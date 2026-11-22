"""Train DMRA on the five selected Madrid months."""
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from dmra import train as core
from dmra.runtime.data_madrid_shift import load_madrid_shift_source, generate_madrid_shift_samples, load_cached_madrid_shift_inputs, ensure_node_prefix
from dmra.madrid_model import model_class, method_name
from dmra.madrid_trainer import run_task

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
    if not np.isclose(float(config["train_rate"]), 0.6, atol=1e-12, rtol=0):
        raise ValueError("Madrid graph availability requires the 60% training split")
    if config["val_test_mix"]:
        raise ValueError("Madrid tasks require chronological validation/test splits")
    if config["pred_dump"]:
        raise ValueError("Legacy prediction_debug exports do not carry Madrid masks; keep pred_dump=false")
    if not 2022<=config["begin_year"]<=config["end_year"]<=2026:
        raise ValueError("Selected-month Madrid task years must lie in2022-2026")
    return config


def task_paths(config, year):
    return (core.project_path(config["raw_data_path"]) / f"{year}.npz",
            core.project_path(config["graph_path"]) / f"{year}_adj.npz")


def load_task(raw_path, graph_path, year, train_rate):
    source = load_madrid_shift_source(raw_path)
    clock = source["timestamps"]
    month_start = clock[0].astype("datetime64[M]")
    month_end = month_start + np.timedelta64(1, "M")
    label = str(month_start)
    if (str(month_start.astype("datetime64[Y]")) != str(year)
            or clock[0] != month_start or clock[-1] + np.timedelta64(15, "m") != month_end):
        raise ValueError(f"Madrid task {year} must cover a complete calendar month in that year")
    ids = source["node_ids"]
    with np.load(graph_path, allow_pickle=False) as archive:
        graph = archive["x"]
        if "node_ids" not in archive or not np.array_equal(archive["node_ids"], ids):
            raise ValueError(f"Madrid graph node IDs differ from source in task {year}")
    if graph.shape != (len(ids), len(ids)) or not np.isfinite(graph).all() or (graph < 0).any():
        raise ValueError(f"Madrid graph is invalid in task {year}")
    active = source["active_nodes"]
    train_stop = int(len(clock) * train_rate)
    if not np.array_equal(active, source["valid"][:train_stop].any(axis=0)):
        raise ValueError(f"Madrid graph availability differs from training observations in task {year}")
    if graph[np.ix_(active, ~active)].any() or graph[np.ix_(~active, active)].any():
        raise ValueError(f"Madrid graph connects training-inactive nodes in task {year}")
    return source, graph, label


def check_data(config):
    previous_ids = None
    tasks = []
    for year in range(config["begin_year"], config["end_year"] + 1):
        source, _, label = load_task(*task_paths(config, year), year, config["train_rate"])
        ids = source["node_ids"]
        if previous_ids is not None:
            ensure_node_prefix(previous_ids, ids)
        previous_ids = ids
        tasks.append({"year": year, "label": label, "nodes": len(ids), "steps": len(source["x"])})
    return tasks


def run(config):
    tasks = check_data(config)
    args=SimpleNamespace(**config)
    for field in ("raw_data_path","graph_path","save_data_path","model_path"):
        setattr(args,field,str(core.project_path(getattr(args,field))))
    args.selected_months = {task["year"]: int(task["label"][-2:]) for task in tasks}
    source_hashes = {}
    stream_digest = hashlib.sha256()
    for year in args.selected_months:
        raw_path, graph_path = task_paths(config, year)
        source_hashes[year] = sha256(raw_path)
        stream_digest.update(bytes.fromhex(source_hashes[year]))
        stream_digest.update(bytes.fromhex(sha256(graph_path)))
    args.stream_sha256 = stream_digest.hexdigest()
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
    (Path(args.path)/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    args.logger.info("device=%s run=%s settings=%s",args.device,args.path,config)
    results={}
    args.graph_size_list=[]
    try:
        for year in range(args.begin_year,args.end_year+1):
            args.year=year
            raw_path, graph_path = task_paths(config, year)
            source, graph, args.task_label = load_task(raw_path, graph_path, year, args.train_rate)
            args.task_month = args.selected_months[year]
            args.source_sha256 = source_hashes[year]
            args.expected_node_ids = source["node_ids"].tolist()
            args.prev_adj = getattr(args, "adj", None)
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
        summary=dict(dataset=DATASET,run=args.path,smoke=args.smoke,tasks=list(results),average=average)
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
