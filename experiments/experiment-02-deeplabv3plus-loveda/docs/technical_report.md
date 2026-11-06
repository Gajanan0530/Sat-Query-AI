# Experiment 02 — DeepLabV3+ on LoveDA

## 1. Experiment Overview

This experiment evaluates DeepLabV3+ with a ResNet-50 encoder
for semantic segmentation of high-resolution remote-sensing imagery.

| Component | Configuration |
|---|---|
| Task | Semantic Segmentation |
| Dataset | LoveDA |
| Architecture | DeepLabV3+ |
| Encoder | ResNet-50 |
| Encoder Pretraining | ImageNet |
| Number of Classes | 7 |
| Original Image Size | 1024 × 1024 |
| Training Patch Size | 512 × 512 |
| Training Epochs | 10 |
| Batch Size | 4 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Loss Function | Cross-Entropy Loss |
| Ignore Index | 255 |
| Precision | FP16 Mixed Precision |
| Hardware | NVIDIA Tesla T4 |

## 2. Dataset

**LoveDA — Land-cover Mapping Dataset**

Official repository:
https://github.com/Junjue-Wang/LoveDA

Dataset source:
https://zenodo.org/records/5706578

The dataset contains high-resolution remote-sensing images
covering urban and rural environments.

### Audited Dataset Split

| Split | Urban | Rural | Total |
|---|---:|---:|---:|
| Training | 1,156 | 1,366 | 2,522 |
| Validation | 677 | 992 | 1,669 |
| Total | 1,833 | 2,358 | 4,191 |

All audited images were 1024 × 1024.
No missing image-mask pairs, dimension mismatches, or corrupt
files were detected in the audited dataset.

### Class Mapping

Original LoveDA mask labels were remapped as follows:

| Original Label | Model Label | Class |
|---:|---:|---|
| 0 | 255 | No-data (ignored) |
| 1 | 0 | Background |
| 2 | 1 | Building |
| 3 | 2 | Road |
| 4 | 3 | Water |
| 5 | 4 | Barren |
| 6 | 5 | Forest |
| 7 | 6 | Agriculture |

No-data pixels were excluded from loss and evaluation.

## 3. Data Preparation

- Images were processed as RGB.
- Each 1024 × 1024 image was divided into four non-overlapping
  512 × 512 quadrants.
- Training selected one random quadrant per original image
  per epoch.
- Validation evaluated all four quadrants deterministically.
- Training augmentation included horizontal flips,
  vertical flips, and 90-degree rotations.
- ImageNet mean and standard deviation were used for
  image normalization.
- Image and mask transformations were synchronized.

### Effective Samples

| Item | Count |
|---|---:|
| Training samples per epoch | 2,522 |
| Training batches per epoch | 630 |
| Validation patches | 6,676 |
| Validation batches | 1,669 |

## 4. Model Architecture

The experiment uses DeepLabV3+ with a ResNet-50 encoder
initialized using ImageNet-pretrained weights.

- Input channels: 3
- Output classes: 7
- Output activation: None (logits)
- Total parameters: 26,679,127

The model was implemented using
`segmentation-models-pytorch`.

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Loss | CrossEntropyLoss |
| Loss Reduction | Sum, normalized by valid pixel count |
| Ignore Index | 255 |
| Scheduler | ReduceLROnPlateau |
| Scheduler Mode | Maximize validation mIoU |
| Scheduler Factor | 0.5 |
| Scheduler Patience | 2 |
| Minimum Learning Rate | 0.000001 |
| Epochs | 10 |
| Batch Size | 4 |
| Mixed Precision | FP16 |

The best checkpoint was selected using validation mIoU.

## 6. Validation Results

Best checkpoint: Epoch 10

| Metric | Result |
|---|---:|
| Validation Loss | 0.86421 |
| Accuracy | 68.30% |
| Mean IoU | 48.21% |
| Mean F1 Score | 63.91% |

### Per-Class Performance

| Class | IoU | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Background | 52.91% | 58.57% | 84.56% | 69.20% |
| Building | 60.12% | 76.30% | 73.92% | 75.09% |
| Road | 50.68% | 85.76% | 55.33% | 67.27% |
| Water | 60.80% | 82.12% | 70.08% | 75.62% |
| Barren | 21.48% | 75.04% | 23.14% | 35.37% |
| Forest | 38.32% | 54.37% | 56.49% | 55.41% |
| Agriculture | 53.19% | 87.27% | 57.67% | 69.44% |

These are validation results, not held-out test-set results.

## 7. Qualitative Evaluation

Four validation images were used for qualitative inspection:

- Urban: 3964.png
- Urban: 4190.png
- Rural: 2522.png
- Rural: 2852.png

The inference pipeline reconstructs a complete 1024 × 1024
prediction from four 512 × 512 patches.

Observed characteristics:

- Large land-cover regions were generally captured.
- Fine boundaries and smaller structures were less consistent.
- Performance varied across rural samples.
- Background confusion was visible in several classes.

The qualitative examples are illustrative and do not replace
the full validation evaluation.

## 8. Standalone Inference

### Requirements

Install the dependencies listed in `requirements.txt`.

### Command

```bash
python inference.py \
    --image "/path/to/input_image.png" \
    --checkpoint "checkpoints/best_deeplabv3plus.pth" \
    --output "inference_output"
```

The input must be an RGB image of size 1024 × 1024.

### Generated Outputs

The standalone inference script generates:

- `prediction_labels.png` — predicted class labels.
- `prediction_color.png` — colorized segmentation map.
- `prediction_overlay.png` — segmentation overlay on the original image.

The standalone inference script was tested successfully
using a LoveDA rural validation image.

## 9. Project Structure

```text
experiment_02/
│
├── dataset/
│   └── LoveDA/
│
├── checkpoints/
│   ├── best_deeplabv3plus.pth
│   └── last_deeplabv3plus.pth
│
├── evaluation/
│   ├── final_evaluation.json
│   ├── final_confusion_matrix.npy
│   ├── confusion_matrix.png
│   ├── class_performance.png
│   └── best_confusion_matrix.npy
│
├── inference/
│   ├── qualitative_evaluation.png
│   ├── qualitative_metrics.json
│   └── standalone_test/
│       ├── prediction_labels.png
│       ├── prediction_color.png
│       └── prediction_overlay.png
│
├── training_logs/
│   ├── training_history.csv
│   └── training_history.json
│
├── dataset_audit.json
├── environment.json
├── experiment_config.json
├── requirements.txt
├── inference.py
└── README.md
```

## 10. Reproducibility and Artifacts

The experiment includes:

- Dataset audit metadata
- Environment and dependency metadata
- Experiment configuration
- Training history in CSV and JSON formats
- Best and last model checkpoints
- Final validation metrics
- Confusion matrix and class performance plots
- Qualitative evaluation outputs
- Standalone inference script

Refer to `experiment_config.json` and `environment.json`
for the recorded experiment settings and software environment.

## 11. Limitations

- Training was limited to a 10-epoch pilot.
- Evaluation was performed on the available validation split.
- No held-out test-set performance is reported.
- Barren and Forest classes showed comparatively weaker IoU.
- Fine structures and class boundaries remain challenging.
- The experiment addresses semantic segmentation rather than
  change detection or object detection.

## 12. Conclusion

This experiment successfully trained and evaluated a
DeepLabV3+ ResNet-50 model on the LoveDA dataset.

The best validation checkpoint achieved 48.21% mIoU and
63.91% mean F1 score after 10 epochs.

The experiment also produced a standalone inference script
that successfully generated label, color, and overlay
predictions from a complete 1024 × 1024 image.
