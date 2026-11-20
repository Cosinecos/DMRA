"""Shared utilities for the Madrid continual forecasting protocol."""

from __future__ import annotations

import numpy as np

GRAPH_KIND = "geodesic_thresholded_gaussian_no_topk_v1"

EARTH_RADIUS_M = 6371000.0

DEFAULT_EPSILON = .01

DEFAULT_RADIUS_M = 4000.0

def _positive(value, name):
    if isinstance(value, bool) or not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)

def distance_matrix_m(coordinates):
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.dtype.kind not in "iuf" or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite numeric (nodes, longitude/latitude) pairs")
    if ((coordinates[:,0] < -180) | (coordinates[:,0] > 180) | (coordinates[:,1] < -90) | (coordinates[:,1] > 90)).any():
        raise ValueError("Coordinates must contain valid longitude/latitude degrees")
    radians = np.deg2rad(coordinates.astype(np.float64))
    lon, lat = radians[:,0], radians[:,1]
    a = np.sin((lat[:,None]-lat[None,:])/2)**2 + np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    result = 2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(a,0,1)))
    np.fill_diagonal(result, 0)
    return result

def _active(active_nodes, nodes):
    active = np.asarray(active_nodes)
    if active.shape != (nodes,) or active.dtype.kind not in "biuf" or not np.isfinite(active).all() or not np.isin(active,(0,1)).all():
        raise ValueError("active_nodes must contain one finite boolean/0-1 flag per node")
    return active.astype(bool)

def sigma_from_first_task(coordinates, active_nodes, candidate_radius_m=DEFAULT_RADIUS_M):
    radius = _positive(candidate_radius_m, "candidate_radius_m")
    distance = distance_matrix_m(coordinates)
    active = _active(active_nodes,len(distance))
    keep = active[:,None] & active[None,:] & (distance>0) & (distance<=radius)
    samples = distance[keep]
    sigma = float(np.std(samples,ddof=0)) if len(samples) else float("nan")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("First task needs nonconstant positive candidate distances to fit sigma; supply --sigma-m explicitly")
    return sigma, dict(fit="first_task_training_active_coordinate_candidates_only",candidate_distance_entries=int(len(samples)),undirected_candidate_pairs=int(len(samples)//2),ddof=0,sigma_m=sigma)

def gaussian_graph(coordinates, active_nodes, sigma_m, epsilon=DEFAULT_EPSILON, candidate_radius_m=DEFAULT_RADIUS_M):
    sigma = _positive(sigma_m,"sigma_m")
    radius = _positive(candidate_radius_m,"candidate_radius_m")
    if isinstance(epsilon,bool) or not np.isfinite(epsilon) or not 0 < float(epsilon) <= 1:
        raise ValueError("epsilon must be finite and in (0,1]")
    distance = distance_matrix_m(coordinates)
    active = _active(active_nodes,len(distance))
    eligible = active[:,None] & active[None,:] & (distance<=radius)
    np.fill_diagonal(eligible,False)
    weights = np.exp(-np.square(distance/sigma))
    graph = np.where(eligible & (weights>=float(epsilon)),weights,0).astype(np.float32)
    empty = graph.sum(axis=1)==0
    index=np.flatnonzero(empty)
    graph[index,index]=1
    return graph
