"""
Model evaluation metrics for both YOLO object detectors and fallback models.
Computes mAP, Precision, Recall, IoU, and inference latency.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import time
from pathlib import Path

import numpy as np
import torch


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2].

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value between 0.0 and 1.0.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate_yolo_detector(
    model: Any,
    data_yaml: str = "data/pcb_defect.yaml",
    split: str = "val",
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "",
) -> Dict[str, Any]:
    """
    Evaluate a YOLO detector using Ultralytics validation API.

    Args:
        model: Ultralytics YOLO model or path to .pt checkpoint.
        data_yaml: Path to dataset YAML configuration.
        split: 'val' or 'test'.
        imgsz: Image resolution.
        batch_size: Batch size for evaluation.
        device: Device ID ('0' or 'cpu').

    Returns:
        Dictionary containing precision, recall, mAP50, mAP50-95, per-class metrics,
        speed (preprocess, inference, postprocess ms), and parameter count.
    """
    from ultralytics import YOLO

    if isinstance(model, (str, Path)):
        model = YOLO(str(model))

    if not device:
        device = 0 if torch.cuda.is_available() else "cpu"

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        verbose=False,
    )

    box_metrics = metrics.box
    class_names = model.names

    per_class = {}
    if hasattr(box_metrics, "p") and hasattr(box_metrics, "r"):
        for i, name in class_names.items():
            if i < len(box_metrics.p) and i < len(box_metrics.r):
                per_class[name] = {
                    "precision": float(box_metrics.p[i]),
                    "recall": float(box_metrics.r[i]),
                    "map50": float(box_metrics.ap50[i]) if hasattr(box_metrics, "ap50") and i < len(box_metrics.ap50) else None,
                }

    speed_info = getattr(metrics, "speed", {})
    inf_speed = speed_info.get("inference", 0.0)

    # Calculate model parameter count
    total_params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") else 0

    return {
        "split": split,
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "map50": float(box_metrics.map50),
        "map50_95": float(box_metrics.map),
        "per_class": per_class,
        "speed_ms": speed_info,
        "inference_latency_ms": inf_speed,
        "model_parameters": total_params,
    }


def evaluate_model(model: Any, test_loader: Any) -> Dict[str, Any]:
    """
    Evaluate a classification model (provided for legacy test harness compatibility).
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    model.eval()
    all_preds, all_labels = [], []
    img_times = []
    start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            t0 = time.time()
            outputs = model(images)
            img_times.append((time.time() - t0) / images.size(0))
            all_preds.extend(outputs.argmax(1).tolist())
            all_labels.extend(labels.tolist())
    total_time = time.time() - start
    n_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0

    return {
        "accuracy": accuracy_score(all_labels, all_preds) if all_labels else 0.0,
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0,
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0,
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist() if all_labels else [],
        "inference_time_s": total_time,
        "avg_image_time_s": sum(img_times) / len(img_times) if img_times else 0.0,
        "model_size_params": n_params,
    }
