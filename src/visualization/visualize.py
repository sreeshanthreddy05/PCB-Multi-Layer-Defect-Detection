"""
Visualization module for PCB defect detection results and ground truth annotations.
Renders predicted bounding boxes, labels, confidence scores, and visual inspection summaries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from src.data.dataset import load_dataset, CLASS_NAMES

DATASET_PATH = "data/raw/pcb-defect-dataset"
OUTPUT_DIR = Path("src/visualization")
NUM_IMAGES = 10

# Color palette for defect classes (RGB normalized 0-1)
CLASS_COLORS = {
    0: "#e6194b",  # mouse_bite - red
    1: "#3cb44b",  # spur - green
    2: "#ffe119",  # missing_hole - yellow
    3: "#4363d8",  # short - blue
    4: "#f58231",  # open_circuit - orange
    5: "#911eb4",  # spurious_copper - purple
}


def render_prediction_image(
    image: Any,
    prediction_result: Dict[str, Any],
) -> bytes:
    """
    Render defect prediction overlay directly to PNG bytes without keeping
    matplotlib figure handles alive in memory.
    """
    import io
    fig = visualize_prediction(image, prediction_result)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_bytes = buf.getvalue()
    plt.close(fig)
    return img_bytes


def visualize_prediction(
    image: Any,
    prediction_result: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Produce a rendered matplotlib figure with predicted bounding boxes, class labels,
    and confidence scores superimposed on the original PCB image.

    Args:
        image: Original PCB image (PIL.Image, np.ndarray, or file path).
        prediction_result: Result dictionary returned by src.inference.predict.predict().
        save_path: Optional path to save the rendered figure.

    Returns:
        Matplotlib Figure object.
    """
    if isinstance(image, (str, Path)):
        img_pil = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img_pil = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
            img_pil = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        else:
            img_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        img_pil = Image.fromarray(np.array(image)).convert("RGB")

    fig = Figure(figsize=(9, 9))
    ax = fig.subplots()
    ax.imshow(img_pil)

    detections = prediction_result.get("detections", [])
    status = prediction_result.get("status", "NO DEFECTS DETECTED")

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            cls_id = det.get("class_id", 0)
            cls_name = det.get("class_name", f"class_{cls_id}")
            conf = det.get("confidence", 0.0)
            color = CLASS_COLORS.get(cls_id, "#ff0000")

            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2.5,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            label_text = f"{cls_name}: {conf:.1f}%"
            ax.text(
                xmin,
                max(12, ymin - 6),
                label_text,
                fontsize=9,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="square,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
            )

    title = f"Inspection: {status} ({len(detections)} defect{'s' if len(detections) != 1 else ''})"
    title_color = "red" if len(detections) > 0 else "green"
    ax.set_title(title, fontsize=12, fontweight="bold", color=title_color)
    ax.axis("off")

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=150)

    return fig



def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH, "train")
    print(f"Loaded {len(dataset)} defect instances.")

    image_paths = list(set(sample[0] for sample in dataset))
    print(f"Found {len(image_paths)} images.")

    random.seed(42)
    selected_images = random.sample(image_paths, min(NUM_IMAGES, len(image_paths)))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {OUTPUT_DIR}")

    for index, image_path in enumerate(selected_images, start=1):
        image = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)

        image_annotations = [sample for sample in dataset if sample[0] == image_path]
        for _, class_id, bbox in image_annotations:
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            rectangle = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rectangle)
            ax.text(
                xmin,
                max(0, ymin - 5),
                CLASS_NAMES[class_id],
                fontsize=9,
                color="red",
                backgroundcolor="white",
            )

        ax.set_title(Path(image_path).name)
        ax.axis("off")

        output_path = OUTPUT_DIR / f"sample_{index}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_path}")

    print("\nDone!")