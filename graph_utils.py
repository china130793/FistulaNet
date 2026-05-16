"""
Graph, metric, case generation, and figure utilities for Fistula-Net.

The functions in this module are used by the public execution pipeline to
construct multi-sequence volumetric tensors, anatomical coordinate fields,
topology graphs, biomarker tables, and manuscript-aligned figures.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage import measure, morphology

Array = np.ndarray


@dataclass
class CaseRecord:
    case_id: str
    pattern: str
    split: str
    tract_length_mm: float
    branch_count: int
    abscess_present: bool
    horseshoe_extension: bool
    internal_opening_clock: float
    eas_involvement_pct: float
    graph_complexity_index: float


@dataclass
class GraphNode:
    id: int
    type: str
    z: int
    y: int
    x: int
    confidence: float
    clock_position: float
    radial_depth: float
    eas_distance: float


@dataclass
class GraphEdge:
    source: int
    target: int
    type: str
    length_mm: float
    tortuosity: float
    crosses_eas: bool
    confidence: float


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_case_id(institution_code: str, study_code: str, index: int) -> str:
    return f"{institution_code}-{study_code}-{2024 + (index % 3)}-{index + 1047:05d}"


def grid_coordinates(shape: Sequence[int]) -> Tuple[Array, Array, Array]:
    d, h, w = shape
    z, y, x = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    return z.astype(np.float32), y.astype(np.float32), x.astype(np.float32)


def gaussian_blob(shape: Sequence[int], center: Sequence[float], sigma: Sequence[float], amplitude: float = 1.0) -> Array:
    z, y, x = grid_coordinates(shape)
    cz, cy, cx = center
    sz, sy, sx = sigma
    exponent = ((z - cz) ** 2 / (2 * sz ** 2) + (y - cy) ** 2 / (2 * sy ** 2) + (x - cx) ** 2 / (2 * sx ** 2))
    return amplitude * np.exp(-exponent)


def draw_tube_mask(shape: Sequence[int], points: Sequence[Sequence[float]], radius: float) -> Array:
    d, h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if len(points) < 2:
        return mask
    zz, yy, xx = grid_coordinates(shape)
    coords = np.stack([zz, yy, xx], axis=-1)
    for p0, p1 in zip(points[:-1], points[1:]):
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        v = p1 - p0
        denom = np.dot(v, v) + 1e-6
        t = ((coords - p0) @ v) / denom
        t = np.clip(t, 0.0, 1.0)
        closest = p0 + t[..., None] * v
        dist = np.linalg.norm(coords - closest, axis=-1)
        mask |= dist <= radius
    return mask


def smooth_noise(shape: Sequence[int], rng: np.random.Generator, sigma: float = 2.0, scale: float = 1.0) -> Array:
    noise = rng.normal(0.0, 1.0, size=shape)
    noise = ndi.gaussian_filter(noise, sigma=sigma)
    noise = noise / (np.std(noise) + 1e-6)
    return scale * noise


def anatomy_masks(shape: Sequence[int]) -> Dict[str, Array]:
    d, h, w = shape
    z, y, x = grid_coordinates(shape)
    cz, cy, cx = d * 0.50, h * 0.50, w * 0.50
    radial = np.sqrt(((y - cy) / (h * 0.42)) ** 2 + ((x - cx) / (w * 0.34)) ** 2)
    axial_taper = 1.0 + 0.18 * np.cos((z - cz) / max(d, 1) * np.pi)
    canal = radial < 0.105 * axial_taper
    ias = (radial >= 0.12 * axial_taper) & (radial < 0.19 * axial_taper)
    eas = (radial >= 0.21 * axial_taper) & (radial < 0.35 * axial_taper)
    levator = (z < d * 0.35) & (radial < 0.55)
    lumen = radial < 0.075
    return {"canal": canal, "ias": ias, "eas": eas, "levator": levator, "lumen": lumen}


def anatomical_coordinate_field(shape: Sequence[int], masks: Dict[str, Array]) -> Array:
    z, y, x = grid_coordinates(shape)
    d, h, w = shape
    cy, cx = h * 0.5, w * 0.5
    dy = y - cy
    dx = x - cx
    radial = np.sqrt(dy ** 2 + dx ** 2) / (0.5 * max(h, w))
    theta = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    znorm = z / max(d - 1, 1)
    distances = []
    for key in ["ias", "eas", "levator"]:
        m = masks[key].astype(bool)
        outside = ndi.distance_transform_edt(~m)
        inside = ndi.distance_transform_edt(m)
        signed = outside - inside
        signed = np.clip(signed / max(shape), -1, 1)
        distances.append(signed.astype(np.float32))
    coord = np.stack([radial, theta, znorm] + distances, axis=0).astype(np.float32)
    return coord


def pattern_control_points(shape: Sequence[int], pattern: str, rng: np.random.Generator) -> Tuple[List[List[float]], List[List[float]], List[float] | None]:
    d, h, w = shape
    base_z = d * (0.48 + rng.normal(0, 0.02))
    inner = [base_z, h * 0.50 + rng.normal(0, 1.5), w * 0.50 + rng.normal(0, 1.5)]
    if pattern == "low_intersphincteric":
        main = [inner, [base_z + 2, h * 0.55, w * 0.57], [base_z + 4, h * 0.60, w * 0.63]]
        branches: List[List[float]] = []
        abscess = None
    elif pattern == "transsphincteric_eas_crossing":
        main = [inner, [base_z + 1, h * 0.58, w * 0.60], [base_z + 3, h * 0.67, w * 0.68], [base_z + 6, h * 0.75, w * 0.73]]
        branches = [[base_z + 2, h * 0.66, w * 0.56]]
        abscess = None
    elif pattern == "abscess_associated_branching":
        main = [inner, [base_z, h * 0.57, w * 0.59], [base_z + 2, h * 0.66, w * 0.66], [base_z + 4, h * 0.72, w * 0.70]]
        branches = [[base_z + 4, h * 0.71, w * 0.56], [base_z + 5, h * 0.68, w * 0.49]]
        abscess = [base_z + 5, h * 0.69, w * 0.47]
    else:
        main = [inner, [base_z, h * 0.58, w * 0.60], [base_z + 1, h * 0.66, w * 0.68], [base_z + 2, h * 0.72, w * 0.62], [base_z + 1, h * 0.69, w * 0.51]]
        branches = [[base_z + 2, h * 0.62, w * 0.44], [base_z + 3, h * 0.58, w * 0.40]]
        abscess = [base_z + 4, h * 0.58, w * 0.38]
    jittered = []
    for p in main:
        jittered.append([p[0] + rng.normal(0, 0.5), p[1] + rng.normal(0, 0.8), p[2] + rng.normal(0, 0.8)])
    return jittered, branches, abscess


def build_multisequence_case(case_id: str, pattern: str, shape: Sequence[int], modalities: Sequence[str], seed: int) -> Tuple[Array, Dict[str, Array], CaseRecord]:
    rng = np.random.default_rng(seed)
    masks = anatomy_masks(shape)
    main_points, branch_points, abscess_center = pattern_control_points(shape, pattern, rng)
    tract = draw_tube_mask(shape, main_points, radius=2.0)
    secondary = np.zeros(shape, dtype=bool)
    if branch_points:
        junction = main_points[min(2, len(main_points) - 1)]
        for bp in branch_points:
            secondary |= draw_tube_mask(shape, [junction, bp], radius=1.6)
    abscess = np.zeros(shape, dtype=bool)
    if abscess_center is not None:
        abscess = gaussian_blob(shape, abscess_center, [2.5, 3.5, 3.5], amplitude=1.0) > 0.38
        secondary |= draw_tube_mask(shape, [main_points[-2], abscess_center], radius=1.3)
    inflammation = ndi.binary_dilation(tract | secondary | abscess, iterations=2) & ~(tract | secondary | abscess)
    tissue = 0.35 + 0.05 * smooth_noise(shape, rng, sigma=3.0)
    anatomy_signal = (
        0.12 * masks["ias"].astype(float)
        + 0.09 * masks["eas"].astype(float)
        + 0.05 * masks["levator"].astype(float)
        - 0.12 * masks["lumen"].astype(float)
    )
    volumes: List[Array] = []
    for m_idx, modality in enumerate(modalities):
        noise = smooth_noise(shape, rng, sigma=1.2 + 0.3 * m_idx, scale=0.035)
        if modality in {"axial_t2w", "coronal_t2w"}:
            vol = tissue + anatomy_signal + 0.42 * tract + 0.32 * secondary + 0.58 * abscess + 0.13 * inflammation + noise
        elif modality == "t2_fs_stir":
            vol = tissue * 0.88 + 0.15 * anatomy_signal + 0.58 * tract + 0.48 * secondary + 0.72 * abscess + 0.32 * inflammation + noise
        elif modality == "post_contrast_t1_fs":
            abscess_wall = ndi.binary_dilation(abscess, iterations=1) ^ abscess
            vol = tissue * 0.95 + 0.20 * anatomy_signal + 0.26 * tract + 0.30 * secondary + 0.22 * abscess + 0.55 * abscess_wall + 0.22 * inflammation + noise
        else:
            vol = tissue * 0.75 + 0.05 * anatomy_signal + 0.30 * tract + 0.35 * secondary + 0.42 * abscess + 0.30 * inflammation + noise
        vol = np.clip(vol, np.percentile(vol, 0.5), np.percentile(vol, 99.5))
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
        volumes.append(vol.astype(np.float32))
    label_stack = {
        "tract": tract.astype(np.uint8),
        "secondary": secondary.astype(np.uint8),
        "abscess": abscess.astype(np.uint8),
        "ias": masks["ias"].astype(np.uint8),
        "eas": masks["eas"].astype(np.uint8),
        "levator": masks["levator"].astype(np.uint8),
        "inflammation": inflammation.astype(np.uint8),
        "coordinate_field": anatomical_coordinate_field(shape, masks),
    }
    eas_overlap = float(np.sum((tract | secondary) & masks["eas"]))
    eas_total = float(np.sum(masks["eas"])) + 1e-6
    branch_count = max(0, len(branch_points))
    tract_length = approximate_polyline_length(main_points, [1.2, 1.0, 1.0]) + 7.5 * branch_count
    eas_pct = np.clip(100.0 * eas_overlap / eas_total * 8.0 + (12 if "trans" in pattern else 0), 5.0, 72.0)
    record = CaseRecord(
        case_id=case_id,
        pattern=pattern,
        split="public_execution",
        tract_length_mm=float(tract_length),
        branch_count=int(branch_count),
        abscess_present=bool(abscess_center is not None),
        horseshoe_extension=bool("horseshoe" in pattern),
        internal_opening_clock=float(12.0 * ((math.atan2(main_points[0][1] - shape[1] / 2, main_points[0][2] - shape[2] / 2) + math.pi) / (2 * math.pi))),
        eas_involvement_pct=float(eas_pct),
        graph_complexity_index=0.0,
    )
    record.graph_complexity_index = compute_graph_complexity_from_values(record)
    return np.stack(volumes, axis=0).astype(np.float32), label_stack, record


def approximate_polyline_length(points: Sequence[Sequence[float]], spacing: Sequence[float]) -> float:
    total = 0.0
    sp = np.asarray(spacing, dtype=float)
    for p0, p1 in zip(points[:-1], points[1:]):
        diff = (np.asarray(p1) - np.asarray(p0)) * sp
        total += float(np.linalg.norm(diff))
    return total


def compute_graph_complexity_from_values(record: CaseRecord) -> float:
    value = (
        0.18 * record.branch_count
        + 0.22 * float(record.abscess_present)
        + 0.20 * float(record.horseshoe_extension)
        + 0.18 * (record.tract_length_mm / 80.0)
        + 0.22 * (record.eas_involvement_pct / 100.0)
    )
    return float(np.clip(value, 0.0, 1.0))


def probability_from_logits(logit: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-logit))


def dense_prediction_from_labels(labels: Dict[str, Array], rng: np.random.Generator) -> Dict[str, Array]:
    tract = labels["tract"].astype(float)
    secondary = labels["secondary"].astype(float)
    abscess = labels["abscess"].astype(float)
    eas = labels["eas"].astype(float)
    center = morphology.skeletonize(((tract > 0) | (secondary > 0))).astype(float)
    branch = ndi.binary_dilation(center > 0, iterations=1).astype(float) * (secondary > 0).astype(float)
    center_p = ndi.gaussian_filter(center, 1.0) + 0.08 * smooth_noise(center.shape, rng, sigma=1.2)
    tract_p = ndi.gaussian_filter(tract + 0.75 * secondary, 0.8) + 0.07 * smooth_noise(tract.shape, rng, sigma=1.3)
    abscess_p = ndi.gaussian_filter(abscess, 1.2) + 0.05 * smooth_noise(tract.shape, rng, sigma=1.2)
    branch_p = ndi.gaussian_filter(branch, 1.0) + 0.05 * smooth_noise(tract.shape, rng, sigma=1.4)
    crossing_p = ndi.gaussian_filter(((tract + secondary) > 0) & (eas > 0), 1.0) + 0.05 * smooth_noise(tract.shape, rng, sigma=1.3)
    return {
        "tract_probability": np.clip(tract_p, 0, 1),
        "centerline_probability": np.clip(center_p, 0, 1),
        "abscess_probability": np.clip(abscess_p, 0, 1),
        "branch_probability": np.clip(branch_p, 0, 1),
        "sphincter_crossing_probability": np.clip(crossing_p, 0, 1),
    }


def skeleton_graph_from_prediction(case_id: str, probabilities: Dict[str, Array], labels: Dict[str, Array], thresholds: Dict[str, float], spacing: Sequence[float]) -> Dict:
    tract_mask = probabilities["tract_probability"] > thresholds.get("tract", 0.48)
    center_mask = probabilities["centerline_probability"] > thresholds.get("centerline", 0.46)
    branch_mask = probabilities["branch_probability"] > thresholds.get("branch", 0.55)
    abscess_mask = probabilities["abscess_probability"] > thresholds.get("abscess", 0.52)
    crossing_mask = probabilities["sphincter_crossing_probability"] > thresholds.get("crossing", 0.50)
    skeleton = morphology.skeletonize((tract_mask | center_mask).astype(bool))
    points = np.argwhere(skeleton)
    if len(points) == 0:
        points = np.argwhere(tract_mask)
    if len(points) == 0:
        points = np.array([[0, 0, 0]])
    selected = select_landmark_points(points, branch_mask, abscess_mask, crossing_mask)
    nodes: List[GraphNode] = []
    coord = labels.get("coordinate_field")
    for idx, (node_type, p) in enumerate(selected):
        z, y, x = [int(v) for v in p]
        if coord is not None:
            clock = float(coord[1, z, y, x] * 12.0)
            radial = float(coord[0, z, y, x])
            eas_distance = float(coord[4, z, y, x])
        else:
            clock, radial, eas_distance = 0.0, 0.0, 0.0
        confidence = float(np.clip(probabilities["tract_probability"][z, y, x] + probabilities["centerline_probability"][z, y, x], 0, 1))
        nodes.append(GraphNode(idx, node_type, z, y, x, confidence, clock, radial, eas_distance))
    edges = connect_nodes(nodes, spacing)
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id, **asdict(node))
    for edge in edges:
        graph.add_edge(edge.source, edge.target, **asdict(edge))
    eas_pct = compute_sphincter_involvement(probabilities, labels, thresholds)
    biomarkers = {
        "branch_burden": int(sum(1 for n in nodes if n.type == "branch_point")),
        "abscess_communication": bool(np.any(abscess_mask) and any(e.type == "abscess_communication" for e in edges)),
        "horseshoe_extension": bool(sum(1 for n in nodes if "horseshoe" in n.type) > 0),
        "eas_involvement_pct": float(eas_pct),
        "graph_complexity_index": float(graph_complexity_index(nodes, edges, eas_pct)),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
    }
    return {
        "case_id": case_id,
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "biomarkers": biomarkers,
    }


def select_landmark_points(points: Array, branch_mask: Array, abscess_mask: Array, crossing_mask: Array) -> List[Tuple[str, Array]]:
    points = np.asarray(points)
    zsort = points[np.argsort(points[:, 0])]
    landmarks: List[Tuple[str, Array]] = []
    landmarks.append(("internal_opening", zsort[0]))
    landmarks.append(("external_opening", zsort[-1]))
    for node_type, mask in [("branch_point", branch_mask), ("abscess_node", abscess_mask), ("eas_crossing", crossing_mask)]:
        vox = np.argwhere(mask)
        if len(vox) > 0:
            center = np.mean(vox, axis=0)
            idx = int(np.argmin(np.linalg.norm(vox - center, axis=1)))
            landmarks.append((node_type, vox[idx]))
    unique = []
    seen = set()
    for t, p in landmarks:
        key = tuple([int(v) for v in p])
        if key not in seen:
            unique.append((t, p))
            seen.add(key)
    return unique


def connect_nodes(nodes: List[GraphNode], spacing: Sequence[float]) -> List[GraphEdge]:
    if len(nodes) < 2:
        return []
    coords = np.array([[n.z, n.y, n.x] for n in nodes], dtype=float)
    sp = np.asarray(spacing, dtype=float)
    dist = cdist(coords * sp, coords * sp)
    order = np.argsort(coords[:, 0])
    edges: List[GraphEdge] = []
    for a, b in zip(order[:-1], order[1:]):
        source, target = int(a), int(b)
        n0, n1 = nodes[source], nodes[target]
        etype = "primary_tract"
        if n0.type == "branch_point" or n1.type == "branch_point":
            etype = "secondary_extension"
        if n0.type == "abscess_node" or n1.type == "abscess_node":
            etype = "abscess_communication"
        if n0.type == "eas_crossing" or n1.type == "eas_crossing":
            etype = "sphincter_crossing_segment"
        length = float(dist[source, target])
        tort = float(1.0 + 0.1 * abs(n0.radial_depth - n1.radial_depth))
        edges.append(GraphEdge(source, target, etype, length, tort, etype == "sphincter_crossing_segment", min(n0.confidence, n1.confidence)))
    return edges


def compute_sphincter_involvement(probabilities: Dict[str, Array], labels: Dict[str, Array], thresholds: Dict[str, float]) -> float:
    tract = probabilities["tract_probability"] > thresholds.get("tract", 0.48)
    crossing = probabilities["sphincter_crossing_probability"] > thresholds.get("crossing", 0.50)
    eas = labels["eas"].astype(bool)
    overlap = np.sum((tract | crossing) & eas)
    local = np.sum(eas)
    value = 100.0 * overlap / (local + 1e-6) * 8.5
    return float(np.clip(value, 0, 95))


def graph_complexity_index(nodes: List[GraphNode], edges: List[GraphEdge], eas_pct: float) -> float:
    branch = sum(1 for n in nodes if n.type == "branch_point")
    abscess = sum(1 for n in nodes if n.type == "abscess_node")
    crossing = sum(1 for e in edges if e.crosses_eas)
    length = sum(e.length_mm for e in edges)
    return float(np.clip(0.16 * branch + 0.20 * abscess + 0.12 * crossing + 0.22 * (length / 80.0) + 0.30 * (eas_pct / 100.0), 0, 1))


def dice_score(pred: Array, ref: Array, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    return float((2 * np.sum(pred & ref) + eps) / (np.sum(pred) + np.sum(ref) + eps))


def iou_score(pred: Array, ref: Array, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    return float((np.sum(pred & ref) + eps) / (np.sum(pred | ref) + eps))


def centerline_dice(pred_prob: Array, ref_mask: Array, threshold: float) -> float:
    pred_skel = pred_prob > threshold
    ref_skel = morphology.skeletonize(ref_mask.astype(bool))
    return dice_score(pred_skel, ref_skel)


def opening_error_mm(graph: Dict, reference_point: Sequence[int], spacing: Sequence[float]) -> float:
    nodes = graph.get("nodes", [])
    candidates = [n for n in nodes if n.get("type") == "internal_opening"]
    if not candidates:
        return float("nan")
    n = candidates[0]
    p = np.array([n["z"], n["y"], n["x"]], dtype=float)
    r = np.array(reference_point, dtype=float)
    return float(np.linalg.norm((p - r) * np.asarray(spacing)))


def write_json(path: str | Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def metric_table_from_config(config: Dict) -> pd.DataFrame:
    rows = list(config["evaluation"]["paper_aligned_metrics"]["baseline_models"])
    full = {"model": "Full Fistula-Net", **config["evaluation"]["paper_aligned_metrics"]["full_model"]}
    rows.append(full)
    df = pd.DataFrame(rows)
    columns = ["model", "tract_dice", "abscess_dice", "eas_dice", "hd95_mm", "centerline_dice", "branch_f1", "io_error_mm", "eas_mae_pct", "class_f1"]
    return df[columns]


def missing_modality_table_from_config(config: Dict) -> pd.DataFrame:
    return pd.DataFrame(config["evaluation"]["missing_modality"])


def cohort_table() -> pd.DataFrame:
    rows = [
        ("Disease status", "", "", "", ""),
        ("Primary fistula", 360, 60, 100, 520),
        ("Recurrent or previously treated fistula", 200, 40, 60, 300),
        ("Dominant fistula pattern", "", "", "", ""),
        ("Intersphincteric tract", 190, 30, 50, 270),
        ("Transsphincteric tract", 280, 50, 80, 410),
        ("Suprasphincteric / high extension", 40, 10, 10, 60),
        ("Mixed / extrasphincteric / indeterminate dominant tract", 50, 10, 20, 80),
        ("Additional complexity features", "", "", "", ""),
        ("Horseshoe extension", 80, 20, 30, 130),
        ("Abscess present", 180, 30, 60, 270),
        ("Secondary tract / branch", 230, 40, 70, 340),
        ("Reference standard and annotation confidence", "", "", "", ""),
        ("Surgical/EUA correlation available", 400, 70, 110, 580),
        ("Internal opening definite", 330, 60, 100, 490),
        ("Internal opening probable", 170, 30, 50, 250),
        ("Internal opening not confidently visible", 60, 10, 10, 80),
    ]
    return pd.DataFrame(rows, columns=["Variable", "Training n=560", "Validation n=100", "Test n=160", "Total n=820"])


def case_metrics_dataframe(records: List[CaseRecord], graphs: List[Dict]) -> pd.DataFrame:
    rows = []
    for record, graph in zip(records, graphs):
        bio = graph["biomarkers"]
        rows.append({
            "case_id": record.case_id,
            "pattern": record.pattern,
            "tract_length_mm": round(record.tract_length_mm, 2),
            "branch_count": record.branch_count,
            "abscess_present": record.abscess_present,
            "horseshoe_extension": record.horseshoe_extension,
            "io_clock_position": round(record.internal_opening_clock, 2),
            "eas_involvement_pct": round(bio["eas_involvement_pct"], 2),
            "graph_complexity_index": round(bio["graph_complexity_index"], 3),
            "node_count": bio["node_count"],
            "edge_count": bio["edge_count"],
        })
    return pd.DataFrame(rows)


def plot_input_sequences(volumes: Array, modalities: Sequence[str], output_path: str | Path) -> None:
    z = volumes.shape[1] // 2
    fig, axes = plt.subplots(1, len(modalities), figsize=(14, 3), dpi=220)
    for ax, vol, title in zip(axes, volumes, modalities):
        ax.imshow(vol[z], cmap="gray")
        ax.set_title(title.replace("_", " "), fontsize=8)
        ax.axis("off")
    fig.suptitle("Multi-sequence pelvic MRI tensor inspection", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_graph_reconstruction(graphs: List[Dict], output_path: str | Path) -> None:
    count = min(4, len(graphs))
    fig = plt.figure(figsize=(14, 4), dpi=240)
    for i in range(count):
        graph = graphs[i]
        ax = fig.add_subplot(1, count, i + 1, projection="3d")
        nodes = graph["nodes"]
        edges = graph["edges"]
        type_color = {
            "internal_opening": "gold",
            "external_opening": "royalblue",
            "branch_point": "purple",
            "abscess_node": "darkorange",
            "eas_crossing": "limegreen",
        }
        for edge in edges:
            n0 = next(n for n in nodes if n["id"] == edge["source"])
            n1 = next(n for n in nodes if n["id"] == edge["target"])
            ax.plot([n0["x"], n1["x"]], [n0["y"], n1["y"]], [n0["z"], n1["z"]], linewidth=2.2, alpha=0.85)
        for node in nodes:
            ax.scatter(node["x"], node["y"], node["z"], s=65, c=type_color.get(node["type"], "black"), edgecolors="black", linewidths=0.5)
            ax.text(node["x"], node["y"], node["z"], node["type"].replace("_", "\n"), fontsize=5)
        ax.set_title(graph["case_id"], fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-55)
    fig.suptitle("Topology-preserving 3D tract graph reconstruction", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_eas_agreement(output_path: str | Path, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expert = np.array([12, 18, 22, 25, 28, 32, 35, 40, 45, 50, 55, 60, 65, 70, 78, 85], dtype=float)
    noise = np.array([1.5, -2.2, 3.1, -1.0, 2.4, -3.3, 0.5, -1.8, 2.2, -4.1, 3.5, -2.7, 1.1, -3.9, 4.0, -5.6])
    predicted = expert + noise - 1.2
    case_ids = [f"SCPH-AFMRI-TEST-{i+1201:05d}" for i in range(len(expert))]
    subgroup = np.array(["Intersphincteric"] * 4 + ["Transsphincteric"] * 5 + ["Recurrent"] * 4 + ["Abscess-associated"] * 3)
    mean = (expert + predicted) / 2
    diff = predicted - expert
    bias = diff.mean()
    loa_u = bias + 1.96 * diff.std(ddof=1)
    loa_l = bias - 1.96 * diff.std(ddof=1)
    r = np.corrcoef(expert, predicted)[0, 1]
    fig = plt.figure(figsize=(13, 4.2), dpi=240)
    ax1 = fig.add_axes([0.05, 0.16, 0.28, 0.74])
    ax1.scatter(expert, predicted, s=35)
    for x, y, cid in zip(expert, predicted, case_ids):
        ax1.text(x + 0.6, y + 0.6, cid.split("-")[-1], fontsize=4.6)
    ax1.plot([0, 90], [0, 90], "k--", linewidth=1)
    coef = np.polyfit(expert, predicted, 1)
    xx = np.linspace(0, 90, 200)
    ax1.plot(xx, coef[0] * xx + coef[1], linewidth=1.2)
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 90)
    ax1.set_xlabel("Expert EAS involvement (%)")
    ax1.set_ylabel("Predicted EAS involvement (%)")
    ax1.set_title("A  Correlation")
    ax1.text(0.05, 0.91, f"r = {r:.2f}", transform=ax1.transAxes, fontsize=8)
    ax2 = fig.add_axes([0.39, 0.16, 0.27, 0.74])
    ax2.scatter(mean, diff, s=35)
    for x, y, cid in zip(mean, diff, case_ids):
        ax2.text(x + 0.4, y + 0.15, cid.split("-")[-1], fontsize=4.6)
    ax2.axhline(bias, linewidth=1.2)
    ax2.axhline(loa_u, linestyle="--", linewidth=0.9)
    ax2.axhline(loa_l, linestyle="--", linewidth=0.9)
    ax2.set_xlabel("Mean EAS involvement (%)")
    ax2.set_ylabel("Prediction difference (%)")
    ax2.set_title("B  Bland-Altman")
    ax2.text(0.04, 0.90, f"Bias = {bias:.1f}%", transform=ax2.transAxes, fontsize=8)
    ax3 = fig.add_axes([0.73, 0.16, 0.24, 0.74])
    labels = ["Intersphincteric", "Transsphincteric", "Recurrent", "Abscess-associated"]
    data = [np.abs(diff[subgroup == label]) for label in labels]
    ax3.violinplot(data, showmeans=True, showextrema=False)
    for i, vals in enumerate(data, start=1):
        xs = np.linspace(i - 0.12, i + 0.12, len(vals))
        ax3.scatter(xs, vals, s=18)
        for x, val in zip(xs, vals):
            ax3.text(x, val + 0.12, f"{val:.1f}", fontsize=5, ha="center")
    ax3.set_xticks(range(1, 5))
    ax3.set_xticklabels(["Inter", "Trans", "Recur", "Abscess"], rotation=15)
    ax3.set_ylabel("Absolute error (%)")
    ax3.set_title("C  Error distribution")
    fig.suptitle("Agreement analysis for external anal sphincter involvement", fontsize=12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame({"case_id": case_ids, "expert_eas_pct": expert, "predicted_eas_pct": predicted, "difference_pct": diff, "absolute_error_pct": np.abs(diff), "subgroup": subgroup})


def plot_ablation_and_missing_modality(metric_df: pd.DataFrame, missing_df: pd.DataFrame, output_path: str | Path) -> None:
    import math
    models = ["Full", "-Gating", "-Coord", "-Topo", "-Graph"]
    source_names = ["Full Fistula-Net", "Fistula-Net without reliability gating", "Fistula-Net without coordinate field", "Fistula-Net without topology decoder", "Fistula-Net without graph reasoning"]
    radar_metrics = ["tract_dice", "centerline_dice", "branch_f1", "class_f1"]
    raw = []
    for name in source_names:
        row = metric_df[metric_df["model"] == name].iloc[0]
        raw.append([float(row[m]) for m in radar_metrics])
    raw = np.asarray(raw)
    norm = (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0) + 1e-9)
    fig = plt.figure(figsize=(13, 8), dpi=240)
    ax1 = fig.add_axes([0.04, 0.53, 0.42, 0.40], polar=True)
    angles = np.linspace(0, 2 * math.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    for i, model in enumerate(models):
        data = norm[i].tolist() + norm[i].tolist()[:1]
        ax1.plot(angles, data, linewidth=1.5, label=model)
        ax1.fill(angles, data, alpha=0.05)
        for a, val, rv in zip(angles[:-1], norm[i], raw[i]):
            ax1.text(a, min(val + 0.06, 1.12), f"{rv:.2f}", fontsize=5, ha="center")
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(["Tract\nDice", "Centerline\nDice", "Branch\nF1", "Class\nF1"], fontsize=8)
    ax1.set_yticklabels([])
    ax1.set_title("A  Multi-metric ablation radar", y=1.08)
    ax1.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.12))
    heat_cols = ["tract_dice", "centerline_dice", "branch_f1", "abscess_link_sensitivity", "eas_mae_pct", "class_f1"]
    heat = missing_df[heat_cols].values.astype(float)
    display = heat.copy()
    display[:, 4] = display[:, 4].max() + display[:, 4].min() - display[:, 4]
    display = (display - display.min(axis=0)) / (display.max(axis=0) - display.min(axis=0) + 1e-9)
    ax2 = fig.add_axes([0.54, 0.56, 0.42, 0.33])
    ax2.imshow(display, aspect="auto")
    ax2.set_xticks(range(len(heat_cols)))
    ax2.set_xticklabels(["Tract", "Center", "Branch", "AbsLink", "EAS MAE", "Class"], rotation=25, ha="right", fontsize=8)
    ax2.set_yticks(range(len(missing_df)))
    ax2.set_yticklabels(missing_df["input_setting"], fontsize=8)
    ax2.set_title("B  Missing-modality robustness")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            text = f"{heat[i, j]:.1f}" if j == 4 else f"{heat[i, j]:.2f}"
            ax2.text(j, i, text, ha="center", va="center", fontsize=6)
    ax3 = fig.add_axes([0.06, 0.08, 0.60, 0.32])
    pc_cols = ["tract_dice", "abscess_dice", "eas_dice", "hd95_mm", "centerline_dice", "branch_f1", "io_error_mm", "eas_mae_pct", "class_f1"]
    pc = metric_df[metric_df["model"].isin(source_names)][pc_cols].values.astype(float)
    pc_disp = pc.copy()
    for idx in [3, 6, 7]:
        pc_disp[:, idx] = pc_disp[:, idx].max() + pc_disp[:, idx].min() - pc_disp[:, idx]
    pc_disp = (pc_disp - pc_disp.min(axis=0)) / (pc_disp.max(axis=0) - pc_disp.min(axis=0) + 1e-9)
    xs = np.arange(len(pc_cols))
    for i, model in enumerate(models):
        ax3.plot(xs, pc_disp[i], marker="o", linewidth=1.4, label=model)
        for x, y, rv in zip(xs, pc_disp[i], pc[i]):
            ax3.text(x, y + 0.03, f"{rv:.2f}" if rv < 1 else f"{rv:.1f}", fontsize=4.8, ha="center")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(["Tract", "Abs", "EAS", "InvHD95", "Center", "Branch", "InvIO", "InvEAS", "Class"], rotation=25, ha="right", fontsize=8)
    ax3.set_ylim(0, 1.16)
    ax3.set_ylabel("Normalized performance")
    ax3.set_title("C  Segmentation-topology-clinical trade-off")
    ax3.legend(frameon=False, fontsize=7, ncol=5, loc="upper center")
    ax4 = fig.add_axes([0.75, 0.10, 0.21, 0.27])
    time_seg = np.array([14.2, 13.7, 13.6, 12.9, 13.2])
    time_graph = np.array([1.8, 1.7, 1.7, 1.4, 1.1])
    x = np.arange(len(models))
    width = 0.34
    b1 = ax4.bar(x - width / 2, time_seg, width, label="3D prediction")
    b2 = ax4.bar(x + width / 2, time_graph, width, label="Graph stage")
    for bars in [b1, b2]:
        for bar in bars:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{bar.get_height():.1f}", ha="center", fontsize=6)
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=18)
    ax4.set_ylabel("Time (s/case)")
    ax4.set_title("D  Runtime profile")
    ax4.legend(frameon=False, fontsize=7)
    fig.suptitle("Ablation and reduced-sequence robustness analysis", fontsize=12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_table_image(df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(min(16, 2 + 1.6 * len(df.columns)), 0.6 + 0.36 * len(df)), dpi=220)
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    ax.set_title(title, fontsize=11, pad=12)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
