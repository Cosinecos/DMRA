"""Train DMRA on the five selected Madrid months."""
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace
import uuid
import calendar

import numpy as np
import torch

from dmra import train as core
from dmra.madrid_checkpoint import protocol
from dmra.runtime.data_madrid_shift import load_madrid_shift_source, generate_madrid_shift_samples, load_cached_madrid_shift_inputs, ensure_node_prefix
from dmra.madrid_model import model_class, method_name
from dmra.madrid_trainer import run_task
from dmra.runtime.prepare_madrid_gaussian import GRAPH_KIND,gaussian_graph,sigma_from_first_task

ROOT=Path(__file__).resolve().parents[1]
DATASET="MADRID_2022_2026_MMD"


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()


def configuration(cli):
    config=core.configuration(cli,DATASET)
    config.update(dataset=DATASET,method=method_name(config["embedding_variant"]))
    if config["steps_per_day"]!=96:
        raise ValueError("Madrid files use15-minute intervals: steps_per_day must be96")
    if config["val_test_mix"]:
        raise ValueError("Madrid tasks require chronological validation/test splits")
    if config["pred_dump"]:
        raise ValueError("Legacy prediction_debug exports do not carry Madrid masks; keep pred_dump=false")
    if not 2022<=config["begin_year"]<=config["end_year"]<=2026:
        raise ValueError("Selected-month Madrid task years must lie in2022-2026")
    return config


def _asset(directory,selection,key):
    filename=selection[key+"_file"]
    if not isinstance(filename,str) or Path(filename).name!=filename:
        raise ValueError("Selection assets must be direct dataset-relative filenames")
    path=directory/filename
    digest=sha256(path)
    if digest!=selection[key+"_sha256"]:
        raise ValueError(f"Madrid selection {key} asset differs from its declared SHA256")
    return json.loads(path.read_text(encoding="utf-8")),digest


def selection_assets(directory,manifest):
    selection=manifest["selection"]
    result,result_sha=_asset(directory,selection,"result")
    protocol_doc,protocol_sha=_asset(directory,selection,"protocol")
    anchor,anchor_sha=_asset(directory,selection,"anchor")
    if result.get("status")!="passed" or result.get("candidate_count")!=58 or result.get("ready_probe_count")!=58:
        raise ValueError("Madrid selection must audit all58 candidate training probes")
    if result.get("selection_protocol_sha256")!=protocol_sha or result.get("anchor_sha256")!=anchor_sha:
        raise ValueError("Madrid selected result does not bind the declared protocol and anchor")
    if result.get("protocol")!="madrid_train_anchor_node_kernel_mmd_v1" or protocol_doc.get("algorithm")!=result["protocol"]:
        raise ValueError("Unexpected Madrid MMD selection algorithm")
    if len(protocol_doc.get("candidates",[]))!=58:
        raise ValueError("Madrid selection protocol must enumerate58 available months")
    ids=np.asarray(anchor["node_ids"])
    if ids.shape!=(128,) or ids.dtype.kind not in "iu" or (ids<0).any() or len(np.unique(ids))!=128:
        raise ValueError("Madrid reference anchor must contain128 unique integer IDs")
    if anchor.get("reference_label")!="2022-01" or anchor.get("seed")!=42:
        raise ValueError("Madrid anchor must be defined from January2022 seed42")
    if not np.array_equal(ids,np.asarray(selection["fixed_reference_node_ids"])):
        raise ValueError("Madrid fixed anchor IDs differ between manifest and reference file")
    expected_labels=selection["selected_labels"]
    if result.get("selected_labels")!=expected_labels or len(expected_labels)!=5:
        raise ValueError("Madrid selected labels differ between manifest and MMD result")
    if len(result.get("selected_months",[]))!=5:
        raise ValueError("Madrid MMD result must contain five selected month records")
    for record,label,year in zip(result["selected_months"],expected_labels,range(2022,2027)):
        if record.get("label")!=label or record.get("year")!=year or label!=f"{year}-{int(record['month']):02d}" or label not in protocol_doc["candidates"]:
            raise ValueError("Madrid MMD result selected month identity is invalid")
    return dict(result=result,anchor=anchor,anchor_ids=ids.astype(np.int64),result_sha256=result_sha,protocol_sha256=protocol_sha,anchor_sha256=anchor_sha)


def task_records(manifest):
    records={}
    for row in manifest["tasks"]:
        year,month=row["year"],row["month"]
        if isinstance(year,bool) or isinstance(month,bool) or not isinstance(year,int) or not isinstance(month,int):
            raise ValueError("Madrid task year/month must be integers")
        if year in records or year not in range(2022,2027) or not 1<=month<=12 or (year==2026 and month>10):
            raise ValueError("Madrid tasks must cover2022-2026 available months, with2026 no later thanOctober")
        if row.get("label")!=f"{year}-{month:02d}":
            raise ValueError("Madrid task label differs from its year/month")
        records[year]=row
    if set(records)!=set(range(2022,2027)):
        raise ValueError("Madrid selected data must declare exactly five2022-2026 tasks")
    return records


def check_data(config):
    manifest_path=core.project_path(config["raw_data_path"]).parent/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset")!=DATASET or manifest.get("format_version")!=1:
        raise ValueError("Unexpected Madrid data manifest")
    if not np.isclose(float(config["train_rate"]),float(manifest["policy"]["train_rate"]),atol=1e-12,rtol=0):
        raise ValueError("Madrid train_rate determined node selection; rebuild a separate cohort before changing it")
    records=task_records(manifest)
    assets=selection_assets(manifest_path.parent,manifest)
    if [records[year]["label"] for year in sorted(records)]!=manifest["selection"]["selected_labels"]:
        raise ValueError("Madrid task months differ from the frozen MMD selection")
    policy=manifest["policy"]["graph"]
    if policy.get("kind")!=GRAPH_KIND or policy.get("distance")!="haversine_geodesic_m_not_routed_road_network_distance" or policy.get("topk") is not None:
        raise ValueError("Madrid MMD tasks require the declared Gaussian geographic prior without top-k")
    if policy.get("sigma_temporal_policy")!="first_selected_task_only_then_frozen":
        raise ValueError("Madrid Gaussian scale must be fitted only on first selected task and frozen")
    sigma,epsilon,radius=float(policy["sigma_m"]),float(policy["epsilon"]),float(policy["candidate_radius_m"])
    first_source=None
    previous=None
    rows=[]
    for year in range(2022,2027):
        if year not in records:
            raise ValueError(f"Madrid manifest has no task {year}")
        raw_path=core.project_path(config["raw_data_path"])/f"{year}.npz"
        graph_path=core.project_path(config["graph_path"])/f"{year}_adj.npz"
        if sha256(raw_path)!=records[year]["raw_file"]["sha256"] or sha256(graph_path)!=records[year]["graph_file"]["sha256"]:
            raise ValueError(f"Madrid source/graph differs from its declared manifest in {year}")
        source=load_madrid_shift_source(raw_path)
        if not np.array_equal(source["node_ids"][:128],assets["anchor_ids"]):
            raise ValueError("Madrid source node registry must preserve the fixed128 anchor prefix")
        month=records[year]["month"]
        steps=calendar.monthrange(year,month)[1]*96
        expected_clock=np.datetime64(f"{year}-{month:02d}-01T00:00:00","ns")+np.arange(steps)*np.timedelta64(15,"m")
        if not np.array_equal(source["timestamps"],expected_clock):
            raise ValueError("Madrid source does not exactly cover its declared complete calendar month")
        if records[year].get("source_steps",steps)!=steps:
            raise ValueError("Madrid manifest source_steps differs from its calendar month")
        if previous is not None:
            ensure_node_prefix(previous,source["node_ids"])
        previous=source["node_ids"]
        with np.load(graph_path,allow_pickle=False) as a:
            graph=a["x"]
            if "node_ids" not in a or not np.array_equal(a["node_ids"],source["node_ids"]):
                raise ValueError(f"Madrid graph node IDs do not match source registry in {year}")
        if graph.shape!=(len(previous),len(previous)) or not np.isfinite(graph).all() or (graph<0).any():
            raise ValueError(f"Madrid graph/source mismatch in task {year}")
        active=source["active_nodes"]
        training_stop=int(len(source["x"])*float(config["train_rate"]))
        if not np.array_equal(active,source["valid"][:training_stop].any(0)):
            raise ValueError("Madrid graph availability must be derived only from this task's training observations")
        if graph[np.ix_(active,~active)].any() or graph[np.ix_(~active,active)].any():
            raise ValueError("Unavailable training nodes must not exchange physical-graph messages")
        expected_graph=gaussian_graph(source["coordinates"],active,sigma,epsilon,radius)
        if not np.array_equal(graph,expected_graph):
            raise ValueError("Madrid source graph differs from its declared Gaussian geometry and parameters")
        if year==2022:
            first_source=source
        rows.append(dict(year=year,month=month,label=records[year]["label"],steps=len(source["x"]),nodes=len(previous),graph_active_nodes=int(active.sum()),observed_labels=int(source["valid"].sum()),observed_zeros=int(((source["x"]==0)&source["valid"]).sum()),dst_invalid_clock_slots=int(source["dst_invalid_clock"].sum())))
    fit=policy["sigma_fit"]
    expected_sigma,details=sigma_from_first_task(first_source["coordinates"],first_source["active_nodes"],radius)
    active_hash=hashlib.sha256(first_source["node_ids"][first_source["active_nodes"]].tobytes()).hexdigest()
    if (fit.get("year")!=2022 or fit.get("month")!=records[2022]["month"] or fit.get("label")!=records[2022]["label"]
            or fit.get("node_ids_sha256")!=active_hash or fit.get("active_nodes")!=int(first_source["active_nodes"].sum())
            or fit.get("candidate_distance_entries")!=details["candidate_distance_entries"] or fit.get("fit")!=details["fit"]
            or not np.isclose(float(fit["sigma_m"]),sigma,rtol=1e-12,atol=0) or not np.isclose(expected_sigma,sigma,rtol=1e-12,atol=0)):
        raise ValueError("Madrid Gaussian bandwidth must bind only the first selected2022 task")
    return [row for row in rows if config["begin_year"]<=row["year"]<=config["end_year"]]


def run(config):
    check_data(config)
    args=SimpleNamespace(**config)
    for field in ("raw_data_path","graph_path","save_data_path","model_path"):
        setattr(args,field,str(core.project_path(getattr(args,field))))
    manifest=Path(args.raw_data_path).parent/"manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Madrid manifest is missing; extract the Madrid dataset archive first")
    args.data_manifest_sha256=sha256(manifest)
    manifest_doc=json.loads(manifest.read_text(encoding="utf-8"))
    records=task_records(manifest_doc)
    assets=selection_assets(manifest.parent,manifest_doc)
    args.selection_result_sha256=assets["result_sha256"]
    args.reference_anchor_sha256=assets["anchor_sha256"]
    args.selected_months={year:record["month"] for year,record in records.items()}
    if args.gpuid>=0:
        if not torch.cuda.is_available() or args.gpuid>=torch.cuda.device_count():
            raise RuntimeError("Requested CUDA device is unavailable; use --gpuid -1 for CPU")
        args.device=torch.device(f"cuda:{args.gpuid}")
    else:
        args.device=torch.device("cpu")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    args.path=str(Path(args.model_path)/f"{stamp}_seed{args.seed}_{uuid.uuid4().hex[:6]}")
    Path(args.path).mkdir(parents=True)
    args.logger=logging.getLogger(args.path)
    args.logger.setLevel(logging.INFO);args.logger.propagate=False
    for handler in (logging.StreamHandler(),logging.FileHandler(Path(args.path)/"train.log",encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"));args.logger.addHandler(handler)
    args.methods={args.method:model_class(args)}
    config=dict(config,data_manifest_sha256=args.data_manifest_sha256,protocol=protocol(args))
    (Path(args.path)/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    args.logger.info("device=%s run=%s settings=%s",args.device,args.path,config)
    results={}
    previous_ids=None
    args.graph_size_list=[]
    try:
        for year in range(args.begin_year,args.end_year+1):
            args.year=year
            args.task_month=records[year]["month"]
            args.task_label=records[year]["label"]
            source_path=Path(args.raw_data_path)/f"{year}.npz"
            source=load_madrid_shift_source(source_path)
            args.source_sha256=sha256(source_path)
            args.expected_node_ids=source["node_ids"].tolist()
            if previous_ids is not None:
                ensure_node_prefix(previous_ids,source["node_ids"])
            previous_ids=source["node_ids"].copy()
            args.prev_adj=getattr(args,"adj",None)
            with np.load(Path(args.graph_path)/f"{year}_adj.npz",allow_pickle=False) as a:
                graph=a["x"]
            args.graph_size=source["x"].shape[1]
            args.expected_source_steps=len(source["x"])
            args.expected_source_first_timestamp=str(source["timestamps"][0])
            args.expected_source_last_timestamp=str(source["timestamps"][-1])
            if year==args.begin_year:
                args.base_node_size=args.graph_size
            args.graph_size_list.append(args.graph_size)
            args.adj=torch.as_tensor(graph/(graph.sum(1,keepdims=True)+1e-6),device=args.device,dtype=torch.float32)
            scope=args.normalization_scope if args.normalize_x else "raw"
            cache=Path(args.save_data_path)/f"explicit_mask_{scope}_L{args.x_len}_H{args.y_len}"/f"{year}.npz"
            if args.data_process or args.smoke:
                output=cache if args.cache_data and not args.smoke else None
                if output is not None:
                    output.parent.mkdir(parents=True,exist_ok=True)
                inputs=generate_madrid_shift_samples(source["x"],source["valid"],source["timestamps"],graph=graph,node_ids=source["node_ids"],active_nodes=source["active_nodes"],coordinates=source.get("coordinates"),dst_invalid_clock=source["dst_invalid_clock"],train_rate=args.train_rate,val_rate=args.val_rate,test_rate=args.test_rate,x_len=args.x_len,y_len=args.y_len,normalize_x=args.normalize_x,normalization_scope=args.normalization_scope,sample_limits={"train":8,"val":4,"test":4} if args.smoke else None,savepath=output,source_sha256=args.source_sha256)
            else:
                inputs=load_cached_madrid_shift_inputs(cache,args)
            args.logger.info("task=%s year=%s source_month=%s registered_nodes=%s training_graph_active=%s samples=%s dst_excluded_windows=%s",args.task_label,year,args.task_month,args.graph_size,int(source["active_nodes"].sum()),{s:len(inputs[f"{s}_x"]) for s in ("train","val","test")},json.loads(str(inputs["__metadata__"]))["dst_excluded_window_counts"])
            results[str(year)]=dict(run_task(inputs,args),month=args.task_month,label=args.task_label)
            (Path(args.path)/"results.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
            del inputs,source
        average={}
        for horizon in results[str(args.begin_year)]["metrics"]:
            average[horizon]={}
            for metric in ("MAE","RMSE","MAPE","WAPE"):
                values=[r["metrics"][horizon][metric] for r in results.values() if r["metrics"][horizon][metric] is not None]
                average[horizon][metric]=float(np.mean(values)) if values else None
        summary=dict(dataset=DATASET,run=args.path,smoke=args.smoke,tasks=list(results),average=average,protocol=protocol(args))
        (Path(args.path)/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        args.logger.info("COMPLETED %s",summary)
        return summary
    finally:
        for handler in list(args.logger.handlers):
            handler.close();args.logger.removeHandler(handler)


def main():
    parser=core.parser()
    parser.description=__doc__
    parser.set_defaults(dataset=DATASET)
    cli=parser.parse_args()
    if cli.dataset.upper().replace("-","_") not in (DATASET,"MADRID","ALL"):
        raise ValueError("This independent entry supports only MADRID_2022_2026_MMD")
    cfg=configuration(cli)
    if cli.check_data:
        rows=check_data(cfg)
        print(json.dumps(dict(status="passed",dataset=DATASET,tasks=rows),indent=2))
        return
    print(json.dumps(dict(status="passed",runs=[run(cfg)]),indent=2))


if __name__=="__main__":
    main()
