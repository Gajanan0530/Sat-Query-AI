import argparse
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp

from PIL import Image


CLASS_NAMES = [
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture"
]

CLASS_COLORS = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [255, 255, 0],
    [0, 0, 255],
    [128, 128, 128],
    [0, 128, 0],
    [255, 165, 0]
], dtype=np.uint8)

PATCH_SIZE = 512
NUM_CLASSES = 7

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def load_model(checkpoint_path, device):

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def preprocess_patch(patch):

    image = patch.astype(np.float32) / 255.0

    image = (image - MEAN) / STD

    image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(
        image.copy()
    ).unsqueeze(0)

    return tensor


@torch.inference_mode()
def predict_image(model, image, device):

    if image.shape != (1024, 1024, 3):
        raise ValueError(
            "Input must be an RGB image of size 1024x1024."
        )

    prediction = np.zeros(
        (1024, 1024),
        dtype=np.uint8
    )

    positions = [
        (0, 0),
        (0, 512),
        (512, 0),
        (512, 512)
    ]

    for top, left in positions:

        patch = image[
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE
        ]

        tensor = preprocess_patch(patch).to(device)

        logits = model(tensor)

        patch_prediction = (
            torch.argmax(logits, dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        prediction[
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE
        ] = patch_prediction

    return prediction


def colorize_prediction(prediction):

    color_mask = CLASS_COLORS[prediction]

    return color_mask


def main():

    parser = argparse.ArgumentParser(
        description="LoveDA DeepLabV3+ inference"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a 1024x1024 RGB image"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_deeplabv3plus.pth"
    )

    parser.add_argument(
        "--output",
        default="inference_output"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    image_array = np.array(image)

    model = load_model(
        args.checkpoint,
        device
    )

    prediction = predict_image(
        model,
        image_array,
        device
    )

    color_mask = colorize_prediction(prediction)

    overlay = (
        0.55 * image_array.astype(np.float32)
        + 0.45 * color_mask.astype(np.float32)
    )

    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    Image.fromarray(prediction).save(
        output_dir / "prediction_labels.png"
    )

    Image.fromarray(color_mask).save(
        output_dir / "prediction_color.png"
    )

    Image.fromarray(overlay).save(
        output_dir / "prediction_overlay.png"
    )

    print("INFERENCE COMPLETED")
    print("Device:", device)
    print("Input:", args.image)
    print("Output directory:", output_dir.resolve())
    print("Generated:")
    print("- prediction_labels.png")
    print("- prediction_color.png")
    print("- prediction_overlay.png")


if __name__ == "__main__":
    main()
