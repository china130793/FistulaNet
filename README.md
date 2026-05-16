# Fistula-Net

Geometry-aware deep learning framework for automated 3D mapping of complex anal fistula tracts and external anal sphincter involvement quantification from multi-sequence pelvic MRI.

This repository contains the public execution package for the Fistula-Net pipeline. The code implements multi-sequence volumetric input handling, sequence-specific 3D encoders, reliability-gated MRI fusion, anatomical coordinate field learning, dual volumetric and topology decoders, topology-preserving graph construction, sphincter-aware graph reasoning, biomarker extraction, and manuscript-aligned table and figure generation.

## Repository contents

```text
Fistula-Net/
├── README.md
├── requirements.txt
├── config.yaml
├── fistulanet_model.py
├── graph_utils.py
├── execute_fistulanet_pipeline.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── graphs/
│   └── snapshots/
└── docs/
    ├── annotation_schema.md
    └── data_availability.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execute pipeline

```bash
python execute_fistulanet_pipeline.py --config config.yaml
```

The command writes the following artifacts:

```text
outputs/figures/figure5_graph_reconstruction.png
outputs/figures/figure6_eas_agreement.png
outputs/figures/figure7_ablation_robustness.png
outputs/figures/input_sequence_tensor_review.png
outputs/tables/cohort_composition_table4.csv
outputs/tables/quantitative_results_table5.csv
outputs/tables/missing_modality_results.csv
outputs/tables/case_level_graph_biomarkers.csv
outputs/tables/figure6_eas_agreement_values.csv
outputs/graphs/SCPH-AFMRI-*-*_graph.json
outputs/snapshots/architecture_integrity_check.json
outputs/snapshots/execution_summary.json
```

## Model workflow

Fistula-Net follows the manuscript methodology:

1. Multi-sequence pelvic MRI tensor preparation.
2. Sequence-specific 3D residual encoding with local axial attention.
3. Reliability-gated fusion to learn local sequence trust weights.
4. Anatomical coordinate field estimation around the anal canal, internal sphincter, external anal sphincter, and levator region.
5. Dual output pathways for volumetric disease mapping and topology prediction.
6. Graph construction using centerline, opening, branch, abscess-link, and sphincter-crossing predictions.
7. Biomarker extraction including internal opening position, branch burden, abscess communication, external anal sphincter involvement, and graph complexity index.

## Privacy and data release

Raw pelvic MRI scans are not included in this repository. The public package provides executable code, generated non-identifiable tensors, graph files, metric tables, figure generation, and documentation. Access to real clinical MRI scans must follow institutional policy, ethics clearance, and controlled data-use agreements.

## Configuration

All execution parameters are stored in `config.yaml`, including volume dimensions, modality names, architecture width, prediction thresholds, output paths, and manuscript-aligned metric values. Modify this file to connect the pipeline to local private MRI tensors.

## Citation

If this repository supports your work, cite the Fistula-Net manuscript and include the repository link.

## Contact

Department of Anorectal Surgery, Shenyang Coloproctology Hospital, Shenyang, Liaoning, China.
