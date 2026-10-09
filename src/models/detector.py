"""
Defect localization/detection support.

Owner: Person 2 (ML Model & Training)

If the dataset provides bounding boxes/masks, this module can host a
proper detector. Otherwise it hosts CAM/Grad-CAM based explainability
localization used to highlight suspected defect regions without
falsely claiming pixel-level segmentation.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, image: torch.Tensor, class_idx: int):
        self.model.zero_grad()
        output = self.model(image)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


def _last_conv(model: nn.Module) -> nn.Module:
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1]


def build_localizer(model: Any, method: str = "gradcam") -> Any:
    """
    Build a localization/explainability wrapper around a trained
    classification model.

    Args:
        model: Trained classification model.
        method: Localization method identifier ("gradcam", "cam",
            "attention", or "bbox" if annotations are available).

    Returns:
        A localizer object exposing a method to generate defect
        region heatmaps/boxes for a given image.
    """
    if method == "bbox":
        return "bbox"
    if method != "gradcam":
        raise NotImplementedError(f"Localization method '{method}' not supported")
    return _GradCAM(model, _last_conv(model))


def get_ground_truth_boxes(dataset: Any, image_path: str) -> list:
    """
    Look up ground-truth defect boxes for an image, from the
    (image_path, class_name, bbox) samples returned by
    src.data.dataset.load_dataset() for the VOC-XML PCB dataset.

    Args:
        dataset: Full sample list from load_dataset().
        image_path: Path of the image to look up.

    Returns:
        List of (class_name, bbox) tuples for that image.
    """
    return [(c, b) for p, c, b in dataset if p == image_path]


def localize_defect(localizer: Any, image: Any, predicted_class: int, gt_boxes: list = None) -> Dict[str, Any]:
    """
    Generate a defect localization result for a single image.

    Per Section 11: when ground-truth annotations are available
    (method="bbox"), use those directly rather than an explainability
    heatmap. Grad-CAM remains the fallback when no annotations exist.

    Args:
        localizer: Object returned by build_localizer().
        image: Preprocessed input image.
        predicted_class: Class index predicted by the classifier.
        gt_boxes: Ground-truth (class_name, bbox) list from
            get_ground_truth_boxes(), required when localizer == "bbox".

    Returns:
        Dictionary containing e.g. {"heatmap": ...} or {"bbox": ..., "all_boxes": ...}
        depending on the method used.
    """
    if localizer == "bbox":
        if not gt_boxes:
            raise ValueError("bbox localization requires gt_boxes")
        return {"bbox": gt_boxes[0][1], "all_boxes": gt_boxes}
    return {"heatmap": localizer.generate(image, predicted_class)}
