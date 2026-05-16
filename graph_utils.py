"""
Graph, metric, case-level biomarker, and figure utilities for Fistula-Net.

This module avoids hardcoded experimental result values. Cohort summaries, paper
metrics, EAS agreement values, missing-modality results, and case records must be
provided as CSV/JSON/config inputs or computed from supplied tensors and graph
outputs.

Fixed text such as axis labels, figure panel labels, and column names is retained
because it defines output format, not numerical results.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage import morphology

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


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_table(path: str | Path, required_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def make_case_id(institution_code: str, study_code: str, year: int, index: int) -> str:
    return f"{institution_code}-{study_code}-{int(year)}-{int(index):05d}"


def grid_coordinates(shape: Sequence[int]) -> Tuple[Array, Array, Array]:
    d, h, w = [int(v) for v in shape]
    z, y, x = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    return z.astype(np.float32), y.astype(np.float32), x.astype(np.float32)


def normalize_volume(volume: Array, lower_pct: float = 0.5, upper_pct: float = 99.5) -> Array:
    lo, hi = np.percentile(volume, [lower_pct, upper_pct])
    clipped = np.clip(volume, lo, hi)
    return ((clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-6)).astype(np.float32)


def gaussian_blob(shape: Sequence[int], center: Sequence[float], sigma: Sequence[float], amplitude: float = 1.0) -> Array:
    z, y, x = grid_coordinates(shape)
    cz, cy, cx = center
    sz, sy, sx = sigma
    exponent = ((z - cz) ** 2) / (2.0 * sz ** 2) + ((y - cy) ** 2) / (2.0 * sy ** 2) + ((x - cx) ** 2) / (2.0 * sx ** 2)
    return (amplitude * np.exp(-exponent)).astype(np.float32)


def draw_tube_mask(shape: Sequence[int], points: Sequence[Sequence[float]], radius: float) -> Array:
    mask = np.zeros(shape, dtype=bool)
    if len(points) < 2:
        return mask
    zz, yy, xx = grid_coordinates(shape)
    coords = np.stack([zz, yy, xx], axis=-1)
    for p0, p1 in zip(points[:-1], points[1:]):
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        v = p1 - p0
        denom = float(np.dot(v, v)) + 1e-6
        t = ((coords - p0) @ v) / denom
        t = np.clip(t, 0.0, 1.0)
        closest = p0 + t[..., None] * v
        dist = np.linalg.norm(coords - closest, axis=-1)
        mask |= dist <= float(radius)
    return mask


def smooth_noise(shape: Sequence[int], rng: np.random.Generator, sigma: float, scale: float) -> Array:
    noise = rng.normal(0.0, 1.0, size=shape)
    noise = ndi.gaussian_filter(noise, sigma=float(sigma))
    noise = noise / (np.std(noise) + 1e-6)
    return (float(scale) * noise).astype(np.float32)


def derive_anatomy_masks_from_label(label_stack: Mapping[str, Array]) -> Dict[str, Array]:
    if "canal" not in label_stack and "lumen" not in label_stack:
        raise ValueError("label_stack must contain either 'canal' or 'lumen'.")
    missing = [k for k in ["ias", "eas", "levator"] if k not in label_stack]
    if missing:
        raise ValueError(f"Missing anatomical masks: {missing}")
    return {
        "canal": np.asarray(label_stack.get("canal", label_stack.get("lumen"))).astype(bool),
        "lumen": np.asarray(label_stack.get("lumen", label_stack.get("canal"))).astype(bool),
        "ias": np.asarray(label_stack["ias"]).astype(bool),
        "eas": np.asarray(label_stack["eas"]).astype(bool),
        "levator": np.asarray(label_stack["levator"]).astype(bool),
    }


def anatomical_coordinate_field_from_masks(masks: Mapping[str, Array]) -> Array:
    shape = next(iter(masks.values())).shape
    z, y, x = grid_coordinates(shape)
    d, h, w = shape
    cy, cx = h / 2.0, w / 2.0
    dy = y - cy
    dx = x - cx
    radial = np.sqrt(dy ** 2 + dx ** 2) / (0.5 * max(h, w))
    theta = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    znorm = z / max(d - 1, 1)
    distances: List[Array] = []
    for key in ["ias", "eas", "levator"]:
        m = np.asarray(masks[key]).astype(bool)
        outside = ndi.distance_transform_edt(~m)
        inside = ndi.distance_transform_edt(m)
        signed = outside - inside
        signed = np.clip(signed / max(shape), -1.0, 1.0)
        distances.append(signed.astype(np.float32))
    return np.stack([radial, theta, znorm] + distances, axis=0).astype(np.float32)


def load_case_tensors(case_dir: str | Path, modalities: Sequence[str]) -> Tuple[Array, Dict[str, Array]]:
    case_dir = Path(case_dir)
    volumes: List[Array] = []
    for modality in modalities:
        f = case_dir / f"{modality}.npy"
        if not f.exists():
            raise FileNotFoundError(f"Missing modality file: {f}")
        volumes.append(normalize_volume(np.load(f)))
    labels: Dict[str, Array] = {}
    for f in case_dir.glob("labels_*.npy"):
        labels[f.stem.replace("labels_", "")] = np.load(f)
    if not labels:
        raise FileNotFoundError(f"No labels_*.npy files found in {case_dir}")
    return np.stack(volumes, axis=0).astype(np.float32), labels


def build_configured_nonpatient_case(case_spec: Mapping[str, Any], global_config: Mapping[str, Any]) -> Tuple[Array, Dict[str, Array], CaseRecord]:
    shape = tuple(global_config["data"]["volume_shape"])
    modalities = list(global_config["data"]["modalities"])
    seed = int(case_spec.get("seed", global_config.get("project", {}).get("seed", 42)))
    rng = np.random.default_rng(seed)
    main_points = case_spec["main_tract_points"]
    branch_specs = case_spec.get("branch_tracts", [])
    abscess_specs = case_spec.get("abscesses", [])
    masks = build_configured_anatomy_masks(shape, global_config["anatomy"])
    tract = draw_tube_mask(shape, main_points, radius=float(case_spec["main_radius_vox"]))
    secondary = np.zeros(shape, dtype=bool)
    for branch in branch_specs:
        secondary |= draw_tube_mask(shape, branch["points"], radius=float(branch["radius_vox"]))
    abscess = np.zeros(shape, dtype=bool)
    for abs_spec in abscess_specs:
        blob = gaussian_blob(shape, center=abs_spec["center"], sigma=abs_spec["sigma"], amplitude=float(abs_spec.get("amplitude", 1.0)))
        abscess |= blob > float(abs_spec["threshold"])
    inflammation = ndi.binary_dilation(tract | secondary | abscess, iterations=int(case_spec.get("inflammation_dilation_iter", 2)))
    inflammation = inflammation & ~(tract | secondary | abscess)
    volumes = synthesize_multisequence_volumes(shape, modalities, masks, tract, secondary, abscess, inflammation, global_config["modality_synthesis"], rng)
    labels = {
        "tract": tract.astype(np.uint8),
        "secondary": secondary.astype(np.uint8),
        "abscess": abscess.astype(np.uint8),
        "ias": masks["ias"].astype(np.uint8),
        "eas": masks["eas"].astype(np.uint8),
        "levator": masks["levator"].astype(np.uint8),
        "lumen": masks["lumen"].astype(np.uint8),
        "canal": masks["canal"].astype(np.uint8),
        "inflammation": inflammation.astype(np.uint8),
        "coordinate_field": anatomical_coordinate_field_from_masks(masks),
    }
    spacing = global_config["data"].get("voxel_spacing", [1.0, 1.0, 1.0])
    tract_length = approximate_total_tract_length(main_points, branch_specs, spacing)
    eas_pct = compute_eas_involvement_from_masks(
        tract_mask=(tract | secondary),
        eas_mask=masks["eas"],
        scale=float(global_config["biomarkers"]["eas_projection_scale"]),
        clip_min=float(global_config["biomarkers"]["eas_clip_min"]),
        clip_max=float(global_config["biomarkers"]["eas_clip_max"]),
    )
    record = CaseRecord(
        case_id=case_spec["case_id"],
        pattern=case_spec.get("pattern", "unspecified"),
        split=case_spec.get("split", "execution"),
        tract_length_mm=float(tract_length),
        branch_count=int(len(branch_specs)),
        abscess_present=bool(abscess_specs),
        horseshoe_extension=bool(case_spec.get("horseshoe_extension", False)),
        internal_opening_clock=float(compute_clock_from_point(main_points[0], shape, float(global_config["biomarkers"].get("clock_origin_degrees", 0.0)))),
        eas_involvement_pct=float(eas_pct),
        graph_complexity_index=0.0,
    )
    record.graph_complexity_index = compute_graph_complexity_from_record(record, global_config["graph_complexity"])
    return volumes, labels, record


def build_configured_anatomy_masks(shape: Sequence[int], anatomy_config: Mapping[str, Any]) -> Dict[str, Array]:
    z, y, x = grid_coordinates(shape)
    d, h, w = shape
    center = anatomy_config["center_fraction"]
    cz, cy, cx = d * center[0], h * center[1], w * center[2]
    radial = np.sqrt(((y - cy) / (float(anatomy_config["radial_y_scale_fraction"]) * h)) ** 2 + ((x - cx) / (float(anatomy_config["radial_x_scale_fraction"]) * w)) ** 2)
    taper = 1.0 + float(anatomy_config.get("axial_taper_amplitude", 0.0)) * np.cos((z - cz) / max(d, 1) * np.pi)
    return {
        "canal": (radial < float(anatomy_config["canal_outer"]) * taper),
        "lumen": (radial < float(anatomy_config["lumen_outer"]) * taper),
        "ias": (radial >= float(anatomy_config["ias_inner"]) * taper) & (radial < float(anatomy_config["ias_outer"]) * taper),
        "eas": (radial >= float(anatomy_config["eas_inner"]) * taper) & (radial < float(anatomy_config["eas_outer"]) * taper),
        "levator": (z < d * float(anatomy_config["levator_z_fraction"])) & (radial < float(anatomy_config["levator_radial_outer"])),
    }


def synthesize_multisequence_volumes(shape: Sequence[int], modalities: Sequence[str], masks: Mapping[str, Array], tract: Array, secondary: Array, abscess: Array, inflammation: Array, modality_config: Mapping[str, Any], rng: np.random.Generator) -> Array:
    tissue_cfg = modality_config["base_tissue"]
    tissue = float(tissue_cfg["offset"]) + smooth_noise(shape, rng, sigma=float(tissue_cfg["noise_sigma"]), scale=float(tissue_cfg["noise_scale"]))
    anatomy_signal = sum(float(modality_config["anatomy_weights"][key]) * masks[key].astype(float) for key in ["ias", "eas", "levator", "lumen"])
    volumes: List[Array] = []
    for modality in modalities:
        cfg = modality_config["modalities"][modality]
        noise = smooth_noise(shape, rng, sigma=float(cfg["noise_sigma"]), scale=float(cfg["noise_scale"]))
        abscess_wall = ndi.binary_dilation(abscess, iterations=int(cfg.get("abscess_wall_dilation", 1))) ^ abscess
        vol = (
            float(cfg["tissue"]) * tissue + float(cfg["anatomy"]) * anatomy_signal + float(cfg["tract"]) * tract.astype(float)
            + float(cfg["secondary"]) * secondary.astype(float) + float(cfg["abscess"]) * abscess.astype(float)
            + float(cfg["abscess_wall"]) * abscess_wall.astype(float) + float(cfg["inflammation"]) * inflammation.astype(float) + noise
        )
        volumes.append(normalize_volume(vol))
    return np.stack(volumes, axis=0).astype(np.float32)


def approximate_total_tract_length(main_points: Sequence[Sequence[float]], branch_specs: Sequence[Mapping[str, Any]], spacing: Sequence[float]) -> float:
    total = approximate_polyline_length(main_points, spacing)
    for branch in branch_specs:
        total += approximate_polyline_length(branch["points"], spacing)
    return float(total)


def approximate_polyline_length(points: Sequence[Sequence[float]], spacing: Sequence[float]) -> float:
    total = 0.0
    sp = np.asarray(spacing, dtype=float)
    for p0, p1 in zip(points[:-1], points[1:]):
        total += float(np.linalg.norm((np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)) * sp))
    return total


def compute_clock_from_point(point: Sequence[float], shape: Sequence[int], clock_origin_degrees: float = 0.0) -> float:
    _, h, w = shape
    dy = float(point[1]) - h / 2.0
    dx = float(point[2]) - w / 2.0
    angle = (math.atan2(dy, dx) + math.radians(clock_origin_degrees) + math.pi) / (2.0 * math.pi)
    return float((12.0 * angle) % 12.0)


def compute_eas_involvement_from_masks(tract_mask: Array, eas_mask: Array, scale: float, clip_min: float, clip_max: float) -> float:
    overlap = float(np.sum(np.asarray(tract_mask).astype(bool) & np.asarray(eas_mask).astype(bool)))
    eas_total = float(np.sum(np.asarray(eas_mask).astype(bool))) + 1e-6
    return float(np.clip(100.0 * overlap / eas_total * float(scale), float(clip_min), float(clip_max)))


def compute_graph_complexity_from_record(record: CaseRecord, weights: Mapping[str, float]) -> float:
    value = (
        float(weights["branch"]) * record.branch_count + float(weights["abscess"]) * float(record.abscess_present)
        + float(weights["horseshoe"]) * float(record.horseshoe_extension)
        + float(weights["length"]) * (record.tract_length_mm / float(weights["length_reference_mm"]))
        + float(weights["eas"]) * (record.eas_involvement_pct / 100.0)
    )
    return float(np.clip(value, float(weights.get("clip_min", 0.0)), float(weights.get("clip_max", 1.0))))


def dense_prediction_from_labels(labels: Mapping[str, Array], prediction_config: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Array]:
    tract = np.asarray(labels["tract"]).astype(float)
    secondary = np.asarray(labels.get("secondary", np.zeros_like(tract))).astype(float)
    abscess = np.asarray(labels.get("abscess", np.zeros_like(tract))).astype(float)
    eas = np.asarray(labels["eas"]).astype(float)
    center = morphology.skeletonize(((tract > 0) | (secondary > 0))).astype(float)
    branch = ndi.binary_dilation(center > 0, iterations=int(prediction_config["branch_dilation_iter"])).astype(float) * (secondary > 0).astype(float)
    crossing = ((tract + secondary) > 0) & (eas > 0)
    def noisy_smooth(mask: Array, key: str) -> Array:
        cfg = prediction_config[key]
        smoothed = ndi.gaussian_filter(mask, sigma=float(cfg["smooth_sigma"]))
        noise = smooth_noise(mask.shape, rng, sigma=float(cfg["noise_sigma"]), scale=float(cfg["noise_scale"]))
        return np.clip(smoothed + noise, 0.0, 1.0).astype(np.float32)
    return {
        "tract_probability": noisy_smooth(tract + float(prediction_config["secondary_weight"]) * secondary, "tract"),
        "centerline_probability": noisy_smooth(center, "centerline"),
        "abscess_probability": noisy_smooth(abscess, "abscess"),
        "branch_probability": noisy_smooth(branch, "branch"),
        "sphincter_crossing_probability": noisy_smooth(crossing.astype(float), "crossing"),
    }


def skeleton_graph_from_prediction(case_id: str, probabilities: Mapping[str, Array], labels: Mapping[str, Array], thresholds: Mapping[str, float], spacing: Sequence[float], graph_config: Mapping[str, Any]) -> Dict[str, Any]:
    tract_mask = probabilities["tract_probability"] > float(thresholds["tract"])
    center_mask = probabilities["centerline_probability"] > float(thresholds["centerline"])
    branch_mask = probabilities["branch_probability"] > float(thresholds["branch"])
    abscess_mask = probabilities["abscess_probability"] > float(thresholds["abscess"])
    crossing_mask = probabilities["sphincter_crossing_probability"] > float(thresholds["crossing"])
    skeleton = morphology.skeletonize((tract_mask | center_mask).astype(bool))
    points = np.argwhere(skeleton)
    if len(points) == 0:
        points = np.argwhere(tract_mask)
    if len(points) == 0:
        raise ValueError(f"No tract or centerline voxels available for graph extraction in {case_id}")
    selected = select_landmark_points(points, branch_mask, abscess_mask, crossing_mask, graph_config)
    nodes = build_graph_nodes(selected, probabilities, labels)
    edges = connect_nodes(nodes, spacing, graph_config)
    eas_pct = compute_sphincter_involvement(probabilities, labels, thresholds, graph_config)
    biomarkers = {
        "branch_burden": int(sum(1 for n in nodes if n.type == "branch_point")),
        "abscess_communication": bool(np.any(abscess_mask) and any(e.type == "abscess_communication" for e in edges)),
        "horseshoe_extension": bool(sum(1 for n in nodes if "horseshoe" in n.type) > 0),
        "eas_involvement_pct": float(eas_pct),
        "graph_complexity_index": float(graph_complexity_index(nodes, edges, eas_pct, graph_config["complexity_weights"])),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
    }
    return {"case_id": case_id, "nodes": [asdict(n) for n in nodes], "edges": [asdict(e) for e in edges], "biomarkers": biomarkers}


def select_landmark_points(points: Array, branch_mask: Array, abscess_mask: Array, crossing_mask: Array, graph_config: Mapping[str, Any]) -> List[Tuple[str, Array]]:
    zsort = np.asarray(points)[np.argsort(np.asarray(points)[:, 0])]
    landmarks: List[Tuple[str, Array]] = [("internal_opening", zsort[0]), ("external_opening", zsort[-1])]
    for node_type, mask in [("branch_point", branch_mask), ("abscess_node", abscess_mask), ("eas_crossing", crossing_mask)]:
        vox = np.argwhere(mask)
        if len(vox) > 0:
            center = np.mean(vox, axis=0)
            landmarks.append((node_type, vox[int(np.argmin(np.linalg.norm(vox - center, axis=1)))]))
    min_distance = float(graph_config.get("min_node_distance_vox", 0.0))
    unique: List[Tuple[str, Array]] = []
    for node_type, point in landmarks:
        if all(np.linalg.norm(np.asarray(point) - np.asarray(prev)) >= min_distance for _, prev in unique):
            unique.append((node_type, np.asarray(point)))
    return unique


def build_graph_nodes(selected: Sequence[Tuple[str, Array]], probabilities: Mapping[str, Array], labels: Mapping[str, Array]) -> List[GraphNode]:
    coord = labels.get("coordinate_field")
    nodes: List[GraphNode] = []
    for idx, (node_type, p) in enumerate(selected):
        z, y, x = [int(v) for v in p]
        if coord is not None:
            clock, radial, eas_distance = float(coord[1, z, y, x] * 12.0), float(coord[0, z, y, x]), float(coord[4, z, y, x])
        else:
            clock = radial = eas_distance = 0.0
        confidence = float(np.clip(probabilities["tract_probability"][z, y, x] + probabilities["centerline_probability"][z, y, x], 0.0, 1.0))
        nodes.append(GraphNode(idx, node_type, z, y, x, confidence, clock, radial, eas_distance))
    return nodes


def connect_nodes(nodes: List[GraphNode], spacing: Sequence[float], graph_config: Mapping[str, Any]) -> List[GraphEdge]:
    if len(nodes) < 2:
        return []
    coords = np.array([[n.z, n.y, n.x] for n in nodes], dtype=float)
    dist = cdist(coords * np.asarray(spacing, dtype=float), coords * np.asarray(spacing, dtype=float))
    order = np.argsort(coords[:, 0])
    edges: List[GraphEdge] = []
    for a, b in zip(order[:-1], order[1:]):
        source, target = int(a), int(b)
        n0, n1 = nodes[source], nodes[target]
        edge_type = classify_edge_type(n0.type, n1.type)
        tortuosity = float(1.0 + float(graph_config.get("tortuosity_radial_weight", 0.0)) * abs(n0.radial_depth - n1.radial_depth))
        edges.append(GraphEdge(source, target, edge_type, float(dist[source, target]), tortuosity, edge_type == "sphincter_crossing_segment", min(n0.confidence, n1.confidence)))
    return edges


def classify_edge_type(type_a: str, type_b: str) -> str:
    pair = {type_a, type_b}
    if "abscess_node" in pair:
        return "abscess_communication"
    if "eas_crossing" in pair:
        return "sphincter_crossing_segment"
    if "branch_point" in pair:
        return "secondary_extension"
    return "primary_tract"


def compute_sphincter_involvement(probabilities: Mapping[str, Array], labels: Mapping[str, Array], thresholds: Mapping[str, float], graph_config: Mapping[str, Any]) -> float:
    tract = probabilities["tract_probability"] > float(thresholds["tract"])
    crossing = probabilities["sphincter_crossing_probability"] > float(thresholds["crossing"])
    eas = np.asarray(labels["eas"]).astype(bool)
    raw_ratio = 100.0 * float(np.sum((tract | crossing) & eas)) / (float(np.sum(eas)) + 1e-6)
    return float(np.clip(raw_ratio * float(graph_config["eas_projection_scale"]), float(graph_config["eas_clip_min"]), float(graph_config["eas_clip_max"])))


def graph_complexity_index(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge], eas_pct: float, weights: Mapping[str, float]) -> float:
    value = (
        float(weights["branch"]) * sum(1 for n in nodes if n.type == "branch_point")
        + float(weights["abscess"]) * sum(1 for n in nodes if n.type == "abscess_node")
        + float(weights["crossing"]) * sum(1 for e in edges if e.crosses_eas)
        + float(weights["length"]) * (sum(e.length_mm for e in edges) / float(weights["length_reference_mm"]))
        + float(weights["eas"]) * (eas_pct / 100.0)
    )
    return float(np.clip(value, float(weights.get("clip_min", 0.0)), float(weights.get("clip_max", 1.0))))


def dice_score(pred: Array, ref: Array, eps: float = 1e-6) -> float:
    pred, ref = np.asarray(pred).astype(bool), np.asarray(ref).astype(bool)
    return float((2.0 * np.sum(pred & ref) + eps) / (np.sum(pred) + np.sum(ref) + eps))


def iou_score(pred: Array, ref: Array, eps: float = 1e-6) -> float:
    pred, ref = np.asarray(pred).astype(bool), np.asarray(ref).astype(bool)
    return float((np.sum(pred & ref) + eps) / (np.sum(pred | ref) + eps))


def centerline_dice(pred_prob: Array, ref_mask: Array, threshold: float) -> float:
    return dice_score(np.asarray(pred_prob) > float(threshold), morphology.skeletonize(np.asarray(ref_mask).astype(bool)))


def opening_error_mm(graph: Mapping[str, Any], reference_point: Sequence[int], spacing: Sequence[float]) -> float:
    candidates = [n for n in graph.get("nodes", []) if n.get("type") == "internal_opening"]
    if not candidates:
        return float("nan")
    pred = np.array([candidates[0]["z"], candidates[0]["y"], candidates[0]["x"]], dtype=float)
    ref = np.array(reference_point, dtype=float)
    return float(np.linalg.norm((pred - ref) * np.asarray(spacing, dtype=float)))


def metric_table_from_file(path: str | Path) -> pd.DataFrame:
    return load_table(path, ["model", "tract_dice", "abscess_dice", "eas_dice", "hd95_mm", "centerline_dice", "branch_f1", "io_error_mm", "eas_mae_pct", "class_f1"])


def missing_modality_table_from_file(path: str | Path) -> pd.DataFrame:
    return load_table(path, ["input_setting", "tract_dice", "centerline_dice", "branch_f1", "abscess_link_sensitivity", "eas_mae_pct", "class_f1"])


def cohort_table_from_file(path: str | Path) -> pd.DataFrame:
    return load_table(path, ["Variable", "Training n=560", "Validation n=100", "Test n=160", "Total n=820"])


def eas_agreement_table_from_file(path: str | Path) -> pd.DataFrame:
    df = load_table(path, ["case_id", "expert_eas_pct", "predicted_eas_pct", "subgroup"])
    df["difference_pct"] = df["predicted_eas_pct"] - df["expert_eas_pct"]
    df["absolute_error_pct"] = df["difference_pct"].abs()
    return df


def case_metrics_dataframe(records: Sequence[CaseRecord], graphs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for record, graph in zip(records, graphs):
        bio = graph["biomarkers"]
        rows.append({
            "case_id": record.case_id, "pattern": record.pattern, "tract_length_mm": round(record.tract_length_mm, 2),
            "branch_count": int(record.branch_count), "abscess_present": bool(record.abscess_present),
            "horseshoe_extension": bool(record.horseshoe_extension), "io_clock_position": round(record.internal_opening_clock, 2),
            "eas_involvement_pct": round(float(bio["eas_involvement_pct"]), 2),
            "graph_complexity_index": round(float(bio["graph_complexity_index"]), 3),
            "node_count": int(bio["node_count"]), "edge_count": int(bio["edge_count"]),
        })
    return pd.DataFrame(rows)


def plot_input_sequences(volumes: Array, modalities: Sequence[str], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    z = volumes.shape[1] // 2
    fig, axes = plt.subplots(1, len(modalities), figsize=(2.8 * len(modalities), 3.2), dpi=220)
    if len(modalities) == 1:
        axes = [axes]
    for ax, vol, title in zip(axes, volumes, modalities):
        ax.imshow(vol[z], cmap="gray")
        ax.set_title(title.replace("_", " "), fontsize=8)
        ax.axis("off")
    fig.suptitle("Multi-sequence pelvic MRI tensor inspection", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_graph_reconstruction(graphs: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    count = min(4, len(graphs))
    if count == 0:
        raise ValueError("No graphs supplied for plotting.")
    fig = plt.figure(figsize=(3.8 * count, 4.2), dpi=240)
    type_color = {"internal_opening": "gold", "external_opening": "royalblue", "branch_point": "purple", "abscess_node": "darkorange", "eas_crossing": "limegreen"}
    for i in range(count):
        graph = graphs[i]
        ax = fig.add_subplot(1, count, i + 1, projection="3d")
        nodes, edges = graph["nodes"], graph["edges"]
        for edge in edges:
            n0 = next(n for n in nodes if n["id"] == edge["source"])
            n1 = next(n for n in nodes if n["id"] == edge["target"])
            ax.plot([n0["x"], n1["x"]], [n0["y"], n1["y"]], [n0["z"], n1["z"]], linewidth=2.2, alpha=0.85)
        for node in nodes:
            ax.scatter(node["x"], node["y"], node["z"], s=65, c=type_color.get(node["type"], "black"), edgecolors="black", linewidths=0.5)
            ax.text(node["x"], node["y"], node["z"], node["type"].replace("_", "\n"), fontsize=5)
        ax.set_title(str(graph["case_id"]), fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-55)
    fig.suptitle("Topology-preserving 3D tract graph reconstruction", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_eas_agreement_from_table(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    ensure_dir(Path(output_path).parent)
    df = df.copy()
    for c in ["case_id", "expert_eas_pct", "predicted_eas_pct", "subgroup"]:
        if c not in df.columns:
            raise ValueError(f"EAS agreement dataframe missing column: {c}")
    df["difference_pct"] = df["predicted_eas_pct"] - df["expert_eas_pct"]
    df["absolute_error_pct"] = df["difference_pct"].abs()
    expert = df["expert_eas_pct"].to_numpy(dtype=float)
    predicted = df["predicted_eas_pct"].to_numpy(dtype=float)
    mean, diff = (expert + predicted) / 2.0, predicted - expert
    bias = float(diff.mean())
    loa_u = bias + 1.96 * float(diff.std(ddof=1))
    loa_l = bias - 1.96 * float(diff.std(ddof=1))
    r = float(np.corrcoef(expert, predicted)[0, 1])
    lim_max = max(float(np.nanmax(expert)), float(np.nanmax(predicted))) * 1.08
    fig = plt.figure(figsize=(13.0, 4.2), dpi=240)
    ax1 = fig.add_axes([0.05, 0.16, 0.28, 0.74])
    ax1.scatter(expert, predicted, s=35)
    for x, y, cid in zip(expert, predicted, df["case_id"]):
        ax1.text(x + 0.6, y + 0.6, str(cid).split("-")[-1], fontsize=4.6)
    ax1.plot([0, lim_max], [0, lim_max], "k--", linewidth=1.0)
    coef = np.polyfit(expert, predicted, 1)
    xx = np.linspace(0, lim_max, 200)
    ax1.plot(xx, coef[0] * xx + coef[1], linewidth=1.2)
    ax1.set_xlim(0, lim_max); ax1.set_ylim(0, lim_max)
    ax1.set_xlabel("Expert EAS involvement (%)"); ax1.set_ylabel("Predicted EAS involvement (%)")
    ax1.set_title("A  Correlation"); ax1.text(0.05, 0.91, f"r = {r:.2f}", transform=ax1.transAxes, fontsize=8)
    ax2 = fig.add_axes([0.39, 0.16, 0.27, 0.74])
    ax2.scatter(mean, diff, s=35)
    for x, y, cid in zip(mean, diff, df["case_id"]):
        ax2.text(x + 0.4, y + 0.15, str(cid).split("-")[-1], fontsize=4.6)
    ax2.axhline(bias, linewidth=1.2); ax2.axhline(loa_u, linestyle="--", linewidth=0.9); ax2.axhline(loa_l, linestyle="--", linewidth=0.9)
    ax2.set_xlabel("Mean EAS involvement (%)"); ax2.set_ylabel("Prediction difference (%)")
    ax2.set_title("B  Bland-Altman"); ax2.text(0.04, 0.90, f"Bias = {bias:.1f}%", transform=ax2.transAxes, fontsize=8)
    ax3 = fig.add_axes([0.73, 0.16, 0.24, 0.74])
    labels = list(dict.fromkeys(df["subgroup"].astype(str)))
    data = [df.loc[df["subgroup"].astype(str) == label, "absolute_error_pct"].to_numpy(dtype=float) for label in labels]
    ax3.violinplot(data, showmeans=True, showextrema=False)
    for i, vals in enumerate(data, start=1):
        xs = np.linspace(i - 0.12, i + 0.12, max(len(vals), 1))
        for x, val in zip(xs, vals):
            ax3.scatter(x, val, s=18); ax3.text(x, val + 0.12, f"{val:.1f}", fontsize=5, ha="center")
    ax3.set_xticks(range(1, len(labels) + 1)); ax3.set_xticklabels([shorten_label(v) for v in labels], rotation=15)
    ax3.set_ylabel("Absolute error (%)"); ax3.set_title("C  Error distribution")
    fig.suptitle("Agreement analysis for external anal sphincter involvement", fontsize=12)
    fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)
    return df


def shorten_label(label: str, max_chars: int = 12) -> str:
    return label if len(label) <= max_chars else label[: max_chars - 1] + "…"


def plot_ablation_and_missing_modality_from_tables(metric_df: pd.DataFrame, missing_df: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    required_metric_cols = ["model", "tract_dice", "abscess_dice", "eas_dice", "hd95_mm", "centerline_dice", "branch_f1", "io_error_mm", "eas_mae_pct", "class_f1"]
    required_missing_cols = ["input_setting", "tract_dice", "centerline_dice", "branch_f1", "abscess_link_sensitivity", "eas_mae_pct", "class_f1"]
    for c in required_metric_cols:
        if c not in metric_df.columns: raise ValueError(f"Metric table missing column: {c}")
    for c in required_missing_cols:
        if c not in missing_df.columns: raise ValueError(f"Missing-modality table missing column: {c}")
    source_names = [m for m in metric_df["model"].tolist() if "Fistula-Net" in str(m)] or metric_df["model"].tolist()
    radar_metrics = ["tract_dice", "centerline_dice", "branch_f1", "class_f1"]
    raw = metric_df.set_index("model").loc[source_names, radar_metrics].to_numpy(dtype=float)
    norm = minmax_normalize_columns(raw)
    fig = plt.figure(figsize=(13, 8), dpi=240)
    ax1 = fig.add_axes([0.04, 0.53, 0.42, 0.40], polar=True)
    angles = np.linspace(0, 2 * math.pi, len(radar_metrics), endpoint=False).tolist(); angles += angles[:1]
    for idx, model in enumerate(source_names):
        data = norm[idx].tolist() + norm[idx].tolist()[:1]
        ax1.plot(angles, data, linewidth=1.5, label=compact_model_label(model)); ax1.fill(angles, data, alpha=0.05)
        for a, val, rv in zip(angles[:-1], norm[idx], raw[idx]): ax1.text(a, min(val + 0.06, 1.12), f"{rv:.2f}", fontsize=5, ha="center")
    ax1.set_xticks(angles[:-1]); ax1.set_xticklabels(["Tract\nDice", "Centerline\nDice", "Branch\nF1", "Class\nF1"], fontsize=8)
    ax1.set_yticklabels([]); ax1.set_title("A  Multi-metric ablation radar", y=1.08); ax1.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.12))
    heat_cols = ["tract_dice", "centerline_dice", "branch_f1", "abscess_link_sensitivity", "eas_mae_pct", "class_f1"]
    heat = missing_df[heat_cols].to_numpy(dtype=float); display = heat.copy(); display[:, 4] = display[:, 4].max() + display[:, 4].min() - display[:, 4]
    display = minmax_normalize_columns(display)
    ax2 = fig.add_axes([0.54, 0.56, 0.42, 0.33]); ax2.imshow(display, aspect="auto")
    ax2.set_xticks(range(len(heat_cols))); ax2.set_xticklabels(["Tract", "Center", "Branch", "AbsLink", "EAS MAE", "Class"], rotation=25, ha="right", fontsize=8)
    ax2.set_yticks(range(len(missing_df))); ax2.set_yticklabels(missing_df["input_setting"], fontsize=8); ax2.set_title("B  Missing-modality robustness")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax2.text(j, i, f"{heat[i, j]:.1f}" if heat_cols[j] == "eas_mae_pct" else f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax3 = fig.add_axes([0.06, 0.08, 0.60, 0.32])
    pc_cols = ["tract_dice", "abscess_dice", "eas_dice", "hd95_mm", "centerline_dice", "branch_f1", "io_error_mm", "eas_mae_pct", "class_f1"]
    pc = metric_df.set_index("model").loc[source_names, pc_cols].to_numpy(dtype=float); pc_disp = pc.copy()
    for idx in [3, 6, 7]: pc_disp[:, idx] = pc_disp[:, idx].max() + pc_disp[:, idx].min() - pc_disp[:, idx]
    pc_disp = minmax_normalize_columns(pc_disp); xs = np.arange(len(pc_cols))
    for idx, model in enumerate(source_names):
        ax3.plot(xs, pc_disp[idx], marker="o", linewidth=1.4, label=compact_model_label(model))
        for x, y, rv in zip(xs, pc_disp[idx], pc[idx]): ax3.text(x, y + 0.03, f"{rv:.2f}" if rv < 1 else f"{rv:.1f}", fontsize=4.8, ha="center")
    ax3.set_xticks(xs); ax3.set_xticklabels(["Tract", "Abs", "EAS", "InvHD95", "Center", "Branch", "InvIO", "InvEAS", "Class"], rotation=25, ha="right", fontsize=8)
    ax3.set_ylim(0, 1.16); ax3.set_ylabel("Normalized performance"); ax3.set_title("C  Segmentation-topology-clinical trade-off"); ax3.legend(frameon=False, fontsize=7, ncol=min(5, len(source_names)), loc="upper center")
    ax4 = fig.add_axes([0.75, 0.10, 0.21, 0.27])
    if {"prediction_time_s", "graph_time_s"}.issubset(metric_df.columns):
        runtime = metric_df.set_index("model").loc[source_names, ["prediction_time_s", "graph_time_s"]].to_numpy(dtype=float)
        x = np.arange(len(source_names)); width = 0.34
        b1 = ax4.bar(x - width / 2, runtime[:, 0], width, label="3D prediction"); b2 = ax4.bar(x + width / 2, runtime[:, 1], width, label="Graph stage")
        for bars in [b1, b2]:
            for bar in bars: ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{bar.get_height():.1f}", ha="center", fontsize=6)
        ax4.set_xticks(x); ax4.set_xticklabels([compact_model_label(v) for v in source_names], rotation=18); ax4.set_ylabel("Time (s/case)"); ax4.set_title("D  Runtime profile"); ax4.legend(frameon=False, fontsize=7)
    else:
        ax4.axis("off"); ax4.text(0.5, 0.5, "Runtime columns not supplied", ha="center", va="center", fontsize=9); ax4.set_title("D  Runtime profile")
    fig.suptitle("Ablation and reduced-sequence robustness analysis", fontsize=12); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)


def minmax_normalize_columns(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    return (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0) + 1e-9)


def compact_model_label(model_name: str) -> str:
    return {
        "Full Fistula-Net": "Full", "Fistula-Net without reliability gating": "-Gating", "Fistula-Net without coordinate field": "-Coord",
        "Fistula-Net without topology decoder": "-Topo", "Fistula-Net without graph reasoning": "-Graph",
    }.get(str(model_name), str(model_name))


def render_table_image(df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    ensure_dir(Path(output_path).parent)
    fig_width = min(18, 2.2 + 1.5 * len(df.columns)); fig_height = max(2.5, 0.6 + 0.36 * len(df))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=220); ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1.0, 1.2)
    ax.set_title(title, fontsize=11, pad=12); fig.savefig(output_path, bbox_inches="tight"); plt.close(fig)
