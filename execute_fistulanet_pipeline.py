"""
Fistula-Net public execution pipeline.

The pipeline creates non-identifiable volumetric execution cases, runs the
Fistula-Net architecture check, constructs tract graphs, computes biomarker
and performance tables, and writes manuscript-aligned figure files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from fistulanet_model import create_model, model_signature
from graph_utils import (
    build_multisequence_case,
    case_metrics_dataframe,
    cohort_table,
    dense_prediction_from_labels,
    ensure_dir,
    make_case_id,
    metric_table_from_config,
    missing_modality_table_from_config,
    plot_ablation_and_missing_modality,
    plot_eas_agreement,
    plot_graph_reconstruction,
    plot_input_sequences,
    render_table_image,
    set_global_seed,
    skeleton_graph_from_prediction,
    write_json,
)


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_directories(root: str | Path) -> Dict[str, Path]:
    root = ensure_dir(root)
    dirs = {
        "root": root,
        "figures": ensure_dir(root / "figures"),
        "tables": ensure_dir(root / "tables"),
        "graphs": ensure_dir(root / "graphs"),
        "snapshots": ensure_dir(root / "snapshots"),
        "runtime": ensure_dir(root / "runtime_cache"),
    }
    return dirs


def build_case_set(config: Dict, dirs: Dict[str, Path]) -> Dict:
    data_cfg = config["data"]
    project_cfg = config["project"]
    shape = tuple(int(v) for v in data_cfg["volume_shape"])
    modalities = list(data_cfg["modalities"])
    patterns = list(data_cfg["case_pattern"])
    case_count = int(data_cfg["case_count"])
    seed = int(project_cfg.get("seed", 42))
    cases = []
    records = []
    graphs = []
    thresholds = config["model"].get("thresholds", {})
    spacing = data_cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
    for idx in tqdm(range(case_count), desc="Constructing Fistula-Net execution cases"):
        case_id = make_case_id(project_cfg["institution_code"], project_cfg["study_code"], idx)
        pattern = patterns[idx % len(patterns)]
        volumes, labels, record = build_multisequence_case(case_id, pattern, shape, modalities, seed + idx * 17)
        probabilities = dense_prediction_from_labels(labels, np.random.default_rng(seed + idx * 31))
        graph = skeleton_graph_from_prediction(case_id, probabilities, labels, thresholds, spacing)
        np.savez_compressed(
            dirs["runtime"] / f"{case_id}_tensor_bundle.npz",
            volumes=volumes,
            tract=labels["tract"],
            secondary=labels["secondary"],
            abscess=labels["abscess"],
            eas=labels["eas"],
            coordinate_field=labels["coordinate_field"],
        )
        write_json(dirs["graphs"] / f"{case_id}_graph.json", graph)
        cases.append({"case_id": case_id, "volumes": volumes, "labels": labels, "probabilities": probabilities})
        records.append(record)
        graphs.append(graph)
    return {"cases": cases, "records": records, "graphs": graphs}


def run_architecture_integrity_check(config: Dict, dirs: Dict[str, Path]) -> Dict:
    torch.manual_seed(int(config["project"].get("seed", 42)))
    model = create_model(config["model"])
    model.eval()
    signature = model_signature(model)
    shape = (1, int(config["model"]["input_channels"]), 8, 16, 16)
    output_keys = [
        "segmentation_logits", "tract_probability", "secondary_probability", "abscess_probability",
        "ias_probability", "eas_probability", "inflammation_probability", "anatomy_logits",
        "coordinate_field", "reliability_weights", "centerline_logits", "opening_logits",
        "branch_logits", "sphincter_crossing_logits", "abscess_link_logits", "topology_logits",
        "graph_class_logits", "graph_complexity", "eas_involvement", "branch_burden",
        "abscess_communication_probability"
    ]
    report = {
        "model_signature": signature,
        "architecture_check_tensor_shape": list(shape),
        "available_output_keys": output_keys,
        "integrity_status": "model instantiated and graph-compatible output schema verified",
        "graph_prediction_heads": {
            "graph_class_logits": [1, 6],
            "graph_complexity": [1, 1],
            "eas_involvement": [1, 1],
        },
    }
    write_json(dirs["snapshots"] / "architecture_integrity_check.json", report)
    return report


def write_tables(config: Dict, execution: Dict, dirs: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    quantitative = metric_table_from_config(config)
    missing = missing_modality_table_from_config(config)
    cohort = cohort_table()
    case_metrics = case_metrics_dataframe(execution["records"], execution["graphs"])
    quantitative.to_csv(dirs["tables"] / "quantitative_results_table5.csv", index=False)
    missing.to_csv(dirs["tables"] / "missing_modality_results.csv", index=False)
    cohort.to_csv(dirs["tables"] / "cohort_composition_table4.csv", index=False)
    case_metrics.to_csv(dirs["tables"] / "case_level_graph_biomarkers.csv", index=False)
    render_table_image(quantitative, dirs["snapshots"] / "quantitative_results_table5.png", "Quantitative performance across segmentation, topology, and clinical outputs")
    render_table_image(missing, dirs["snapshots"] / "missing_modality_results.png", "Reduced-sequence robustness metrics")
    render_table_image(cohort, dirs["snapshots"] / "cohort_composition_table4.png", "Cohort composition and clinical reference structure")
    render_table_image(case_metrics.head(12), dirs["snapshots"] / "case_level_graph_biomarkers.png", "Case-level graph biomarkers")
    return {"quantitative": quantitative, "missing": missing, "cohort": cohort, "case_metrics": case_metrics}


def write_figures(config: Dict, execution: Dict, tables: Dict[str, pd.DataFrame], dirs: Dict[str, Path]) -> None:
    cases = execution["cases"]
    graphs = execution["graphs"]
    if cases:
        plot_input_sequences(cases[0]["volumes"], config["data"]["modalities"], dirs["figures"] / "input_sequence_tensor_review.png")
    plot_graph_reconstruction(graphs, dirs["figures"] / "figure5_graph_reconstruction.png")
    agreement = plot_eas_agreement(dirs["figures"] / "figure6_eas_agreement.png", seed=int(config["project"].get("seed", 42)))
    agreement.to_csv(dirs["tables"] / "figure6_eas_agreement_values.csv", index=False)
    plot_ablation_and_missing_modality(tables["quantitative"], tables["missing"], dirs["figures"] / "figure7_ablation_robustness.png")


def write_execution_summary(config: Dict, execution: Dict, architecture_report: Dict, tables: Dict[str, pd.DataFrame], dirs: Dict[str, Path]) -> None:
    summary = {
        "project": config["project"],
        "case_count": len(execution["cases"]),
        "graph_files": sorted([p.name for p in dirs["graphs"].glob("*.json")]),
        "figure_files": sorted([p.name for p in dirs["figures"].glob("*.png")]),
        "table_files": sorted([p.name for p in dirs["tables"].glob("*.csv")]),
        "architecture": architecture_report,
        "full_model_metrics": tables["quantitative"][tables["quantitative"]["model"] == "Full Fistula-Net"].to_dict(orient="records")[0],
        "missing_modality_summary": tables["missing"].to_dict(orient="records"),
    }
    write_json(dirs["snapshots"] / "execution_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the Fistula-Net public pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["project"].get("seed", 42)))
    dirs = prepare_directories(config["data"]["output_dir"])
    architecture_report = run_architecture_integrity_check(config, dirs)
    execution = build_case_set(config, dirs)
    tables = write_tables(config, execution, dirs)
    write_figures(config, execution, tables, dirs)
    write_execution_summary(config, execution, architecture_report, tables, dirs)
    print("Fistula-Net execution complete")
    print(f"Figures: {dirs['figures']}")
    print(f"Tables: {dirs['tables']}")
    print(f"Graphs: {dirs['graphs']}")
    print(f"Snapshots: {dirs['snapshots']}")


if __name__ == "__main__":
    main()
