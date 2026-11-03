# SatQuery AI

**Deep Learning · Remote Sensing · Satellite Image Analysis**

SatQuery AI is a research-oriented project exploring the application of deep learning and computer vision to satellite imagery and remote sensing.

The project focuses on implementing, training, fine-tuning, and evaluating deep learning architectures across different geospatial image-analysis tasks. Each experiment is organized to document its methodology, dataset configuration, model architecture, training process, evaluation results, and inference capabilities.

---

## Research Experiments

| No. | Model Architecture | Dataset | Research Task | Status |
|---|---|---|---|---|
| 01 | BIT-CD | LEVIR-CD | Change Detection | Completed |
| 02 | DeepLabV3+ (ResNet-50) | LoveDA | Semantic Segmentation | Completed |
| 03 | To be determined | DIOR | Object Detection | Planned |

### Experiment 01 — BIT-CD: Change Detection

- **Architecture:** BIT-CD (`base_transformer_pos_s4_dd8_dedim8`)
- **Dataset:** LEVIR-CD
- **Task:** Binary building change detection using bi-temporal satellite imagery.
- **Input resolution:** 256 × 256
- **Training:** 10 epochs

**Reported test-set results:**

| Metric | Result |
|---|---:|
| Accuracy | 96.67% |
| mIoU | 71.42% |
| mF1 | 80.76% |
| Changed-class F1 | 63.26% |
| Precision | 72.15% |
| Recall | 56.33% |

The experiment demonstrates the use of transformer-based architectures to identify changes between satellite images captured at different points in time.

### Experiment 02 — DeepLabV3+: Semantic Segmentation

- **Architecture:** DeepLabV3+
- **Encoder:** ResNet-50 (ImageNet-pretrained)
- **Dataset:** LoveDA
- **Task:** Seven-class semantic segmentation of satellite imagery.
- **Input image size:** 1024 × 1024
- **Training patch size:** 512 × 512
- **Training:** 10 epochs
- **Optimizer:** AdamW
- **Hardware:** NVIDIA Tesla T4 GPUs

**Validation results:**

| Metric | Result |
|---|---:|
| Accuracy | 68.30% |
| Mean IoU (mIoU) | 48.21% |
| Mean F1-score (mF1) | 63.91% |

**Segmentation classes:**

1. Background
2. Building
3. Road
4. Water
5. Barren
6. Forest
7. Agriculture

The experiment includes dataset auditing, preprocessing, data augmentation, model training, validation, class-wise evaluation, confusion-matrix analysis, qualitative inference, and a standalone inference implementation.

**Evaluation note:** The reported metrics are from the validation dataset. They should not be interpreted as held-out test-set results.

For detailed methodology, configurations, evaluation results, and inference instructions, refer to:

[`Experiment 02 — DeepLabV3+ on LoveDA`](experiments/experiment-02-deeplabv3plus-loveda/README.md)

---

## Repository Structure

```text
Sat-Query-AI/
│
├── README.md
│
├── experiments/
│   │
│   ├── experiment-01-bit-cd/
│   │
│   ├── experiment-02-deeplabv3plus-loveda/
│   │
│   └── experiment-03/
│
└── docs/
```

Each experiment maintains its own documentation, implementation, configurations, evaluation records, and relevant outputs.

---

## Research Methodology

The experiments follow a structured workflow:

1. **Research and Architecture Selection** — Identify relevant research papers and deep learning architectures.
2. **Dataset Preparation** — Acquire, inspect, audit, and organize the datasets.
3. **Preprocessing** — Apply image transformations, normalization, patch extraction, and augmentation where appropriate.
4. **Model Implementation** — Configure the selected architecture and pretrained components.
5. **Training and Fine-tuning** — Train models using documented hyperparameters and computational resources.
6. **Evaluation** — Measure performance using task-appropriate metrics and analyze class-wise results.
7. **Inference and Visualization** — Generate predictions and inspect qualitative outputs.
8. **Documentation** — Record experiment configurations, observations, limitations, and reproducibility requirements.

---

## Technology Stack

- **Programming:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Computer Vision:** OpenCV
- **Model Architectures:** BIT-CD, DeepLabV3+
- **Model Utilities:** Segmentation Models PyTorch
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Development Environment:** Kaggle Notebooks
- **Version Control:** Git and GitHub

The technology stack may vary across individual experiments.

---

## Datasets

The project uses publicly available remote-sensing datasets for different computer-vision tasks.

| Dataset | Intended Task |
|---|---|
| LEVIR-CD | Change Detection |
| LoveDA | Land-Cover Semantic Segmentation |
| DIOR | Object Detection |

Original datasets are not included in this repository. Refer to the respective experiment documentation for dataset sources, preparation instructions, and directory requirements.

---

## Reproducibility and Model Artifacts

Each experiment aims to provide the materials necessary to understand and reproduce its methodology.

Depending on the experiment, these may include:

- Training and evaluation notebooks
- Model configuration files
- Dependency specifications
- Training logs
- Evaluation metrics
- Confusion matrices and performance visualizations
- Standalone inference scripts
- Experiment reports

Large datasets and model checkpoints may be hosted separately rather than committed directly to GitHub.

---

## Current Project Scope

SatQuery AI currently explores three major areas of satellite-image analysis:

- **Change Detection:** Identifying changes between multi-temporal satellite images.
- **Semantic Segmentation:** Assigning meaningful land-cover categories to individual image pixels.
- **Object Detection:** Identifying and localizing objects in satellite imagery.

The project is experimental and research-oriented. Results are reported according to the evaluation procedures used in each individual experiment.

---

## License

No open-source license has been assigned yet. Until a license is added, the repository's contents remain subject to the applicable default copyright restrictions.
