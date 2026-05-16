# Fistula-Net

**Fistula-Net** is a geometry-aware deep learning framework for automated 3D mapping of complex anal fistula tracts and external anal sphincter involvement quantification from multi-sequence pelvic MRI.

This repository contains the public execution package for the Fistula-Net pipeline. The code implements multi-sequence volumetric input handling, sequence-specific 3D encoders, reliability-gated MRI fusion, anatomical coordinate field learning, dual volumetric and topology decoders, topology-preserving graph construction, sphincter-aware graph reasoning, biomarker extraction, and manuscript-aligned table and figure generation.

## Authors

**Xiaopeng Wang, Hexue Yuan, Qian Xu**

Department of Anorectal Surgery, Shenyang Coloproctology Hospital, Shenyang, Liaoning 110001, China.

**Corresponding author:**  
Xiaopeng Wang  
Department of Anorectal Surgery, Shenyang Coloproctology Hospital, Shenyang, Liaoning 110001, China.  
Email: 17702466890@163.com

**ORCID:**  
Xiaopeng Wang: 0009-0001-2872-3396  
Qian Xu: 0009-0007-6003-2193

## Repository Contents

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

Create a virtual environment and install the required packages.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execute Pipeline

Run the main execution script:

```bash
python execute_fistulanet_pipeline.py --config config.yaml
```

The command writes the following artifacts:

```text
outputs/figures/figure5_graph_reconstruction.png
outputs/figures/figure6_eas_agreement.png
outputs/figures/figure7_ablation_robustness.png
outputs/tables/cohort_composition_table4.csv
outputs/tables/quantitative_results_table5.csv
outputs/tables/missing_modality_results.csv
outputs/tables/case_level_graph_biomarkers.csv
outputs/tables/figure6_eas_agreement_values.csv
outputs/graphs/SCPH-AFMRI-*-*_graph.json
outputs/snapshots/architecture_integrity_check.json
outputs/snapshots/execution_summary.json
```

## Model Workflow

Fistula-Net follows the manuscript methodology:

1. Multi-sequence pelvic MRI tensor preparation.
2. Sequence-specific 3D residual encoding with local axial attention.
3. Reliability-gated fusion to learn local sequence trust weights.
4. Anatomical coordinate field estimation around the anal canal, internal anal sphincter, external anal sphincter, and levator region.
5. Dual output pathways for volumetric disease mapping and topology prediction.
6. Graph construction using centerline, opening, branch, abscess-link, and sphincter-crossing predictions.
7. Biomarker extraction including internal opening position, branch burden, abscess communication, external anal sphincter involvement, and graph complexity index.

## Outputs

The repository generates manuscript-aligned outputs for technical review and reproducibility, including:

- 3D tract graph reconstruction figures.
- External anal sphincter involvement agreement plots.
- Ablation and missing-modality robustness plots.
- Cohort composition and quantitative result tables.
- Case-level graph biomarker files.
- De-identified graph JSON outputs.

## Privacy and Data Release

Raw pelvic MRI scans are not included in this repository. Medical imaging data may contain protected health information in DICOM metadata, private scanner tags, acquisition dates, institution identifiers, and image-associated annotations.

The public package provides executable code, generated non-identifiable tensors, graph files, metric tables, figure-generation utilities, and documentation. Access to real clinical MRI scans must follow institutional policy, ethics clearance, and controlled data-use agreements.

## Configuration

All execution parameters are stored in `config.yaml`, including volume dimensions, modality names, architecture width, prediction thresholds, output paths, and manuscript-aligned metric values.

Modify `config.yaml` to connect the pipeline to local private MRI tensors.

## Citation

If this repository supports your work, please cite the Fistula-Net manuscript and include the repository link.

```text
Wang X, Yuan H, Xu Q. Fistula-Net: A Geometry-Aware Deep Learning Framework for Automated 3D Mapping of Complex Anal Fistula Tracts and Sphincter-Involvement Quantification.
```

## Contact

For questions related to this repository or manuscript, contact:

**Xiaopeng Wang**  
Department of Anorectal Surgery  
Shenyang Coloproctology Hospital  
Shenyang, Liaoning 110001, China  
Email: 17702466890@163.com
