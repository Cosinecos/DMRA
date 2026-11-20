"""Shared utilities for the Madrid continual forecasting protocol."""

import hashlib

import json

from pathlib import Path

import calendar

import numpy as np

from madrid_runtime import main as core

from data_madrid_shift import load_madrid_shift_source, ensure_node_prefix

from prepare_madrid_gaussian import GRAPH_KIND, gaussian_graph, sigma_from_first_task

ROOT=Path(__file__).resolve().parents[1]

DATASET="MADRID_2022_2026_MMD"

def sha256(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()

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
