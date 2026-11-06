# Experiment 02 — DeepLabV3+ Semantic Segmentation

**SatQuery AI | Model Training, Evaluation & Inference Report**

## 1. Project Overview

This experiment documents the training, fine-tuning, and evaluation of DeepLabV3+ with a ResNet-50 encoder for semantic segmentation of satellite imagery using the LoveDA dataset.

The model performs pixel-level classification, assigning each pixel to one of seven land-cover categories.

## 2. Dataset and Model Configuration

| Item | Configuration |
|---|---|
| Dataset | LoveDA |
| Task | Semantic Segmentation |
| Model Architecture | DeepLabV3+ |
| Encoder | ResNet-50 |
| Encoder Pretraining | ImageNet |
| Number of Classes | 7 |
| Original Image Size | 1024 × 1024 |
| Training Patch Size | 512 × 512 |
| Training Epochs | 10 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.0001 |
| Loss Function | CrossEntropyLoss |
| Hardware | NVIDIA Tesla T4 GPU |
| Mixed Precision | FP16 |

**Output Classes:**

1. Background
2. Building
3. Road
4. Water
5. Barren
6. Forest
7. Agriculture

## 3. Validation Evaluation Results

The following metrics were obtained using the best-performing checkpoint from the 10-epoch training run.

| Metric | Result |
|---|---:|
| Accuracy | 68.30% |
| Mean Intersection over Union (mIoU) | 48.21% |
| Mean F1-score (mF1) | 63.91% |
| Validation Loss | 0.8642 |

**Per-class IoU:**

| Class | IoU |
|---|---:|
| Background | 52.91% |
| Building | 60.12% |
| Road | 50.68% |
| Water | 60.80% |
| Barren | 21.48% |
| Forest | 38.32% |
| Agriculture | 53.19% |

## 4. Inference Demonstration

The best checkpoint was loaded into a standalone inference pipeline and tested on a LoveDA rural validation image.

The inference script successfully generated:

- `prediction_labels.png` — Pixel-level predicted class labels.
- `prediction_color.png` — Color-coded semantic segmentation map.
- `prediction_overlay.png` — Segmentation predictions overlaid on the original satellite image.

Standalone inference was successfully verified using GPU-based PyTorch execution.

## 5. Outputs Generated

The experiment produced the following artifacts:

- `best_deeplabv3plus.pth` — Best model checkpoint.
- `last_deeplabv3plus.pth` — Final training checkpoint.
- `training_history.csv` — Epoch-wise training and validation logs.
- `final_evaluation.json` — Final validation metrics.
- `confusion_matrix.png` — Normalized confusion matrix.
- `class_performance.png` — Per-class performance visualization.
- `qualitative_evaluation.png` — Qualitative segmentation results.
- `inference.py` — Standalone inference implementation.
- `experiment_config.json` — Experiment configuration.
- `requirements.txt` — Dependency specifications.

## 6. Scope and Limitations

The experiment evaluates seven-class semantic segmentation using the LoveDA validation dataset.

The best validation mIoU was 48.21% after 10 training epochs. Performance varies considerably across classes, with Barren land presenting the most challenging segmentation category.

The reported metrics represent validation performance and should not be interpreted as independent test-set results.

## 7. Handover Package

The handover package includes the standalone inference script, model configuration, dependency specifications, evaluation records, and trained model checkpoints.

The LoveDA dataset is not required for ordinary inference on a new compatible satellite image. However, it is required to reproduce the training and validation experiments.

## Model Weights

The trained model checkpoints are hosted on Hugging Face.

**Hugging Face Repository:** [SatQuery AI — DeepLabV3+ on LoveDA](https://huggingface.co/Gajanan0530/satquery-ai-loveda-deeplabv3plus)

| Checkpoint | Description | Download |
|---|---|---|
| Best | Best validation checkpoint (Epoch 10) | [Download](https://huggingface.co/Gajanan0530/satquery-ai-loveda-deeplabv3plus/resolve/main/best_deeplabv3plus.pth) |
| Last | Final training checkpoint (Epoch 10) | [Download](https://huggingface.co/Gajanan0530/satquery-ai-loveda-deeplabv3plus/resolve/main/last_deeplabv3plus.pth) |

The best checkpoint is recommended for inference and demonstration. Both checkpoints are provided for reproducibility.
---

**SatQuery AI | DeepLabV3+ Semantic Segmentation Report**
