"""
Inference pipeline for PCB Defect Detection.
Supports both YOLO object detection models (returning bounding boxes, confidence, class IDs)
and fallback PyTorch classifiers.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_INFERENCE_LOCK = threading.Lock()

DEFECT_CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}


def _to_tensor(image: Any) -> torch.Tensor:
    x = torch.as_tensor(image).float()
    if x.dim() == 3:
        x = x.permute(2, 0, 1)
    return x.unsqueeze(0)


def predict(
    image: Any,
    model: Any,
    confidence_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Run inference on a single PCB image.

    Args:
        image: PIL Image, np.ndarray, file path, or tensor.
        model: Loaded YOLO model or PyTorch model.
        confidence_threshold: Minimum confidence score to retain detections (default: 0.25).

    Returns:
        Structured inspection result dictionary containing:
            - status: "DEFECTS DETECTED" | "NO DEFECTS DETECTED" | "ERROR"
            - defect_count: int
            - detections: list of dicts [{"class_id": int, "class_name": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]
            - processing_time_ms: float
            - model_name: str
            - class: int (primary defect class ID or None)
            - confidence: float (primary defect confidence in %)
    """
    t0 = time.time()

    # Check if model is an Ultralytics YOLO detector
    if hasattr(model, "predict"):
        # Convert PIL or numpy if needed
        if isinstance(image, str) or isinstance(image, Path):
            img_input = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # If normalized float [0, 1], scale to uint8
            if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
                img_input = (image * 255).astype(np.uint8)
            else:
                img_input = image
        else:
            img_input = image

        with _INFERENCE_LOCK:
            results = model.predict(
                source=img_input,
                conf=confidence_threshold,
                verbose=False,
            )
        elapsed_ms = (time.time() - t0) * 1000

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                cls_name = model.names.get(cls_id, DEFECT_CLASS_NAMES.get(cls_id, f"class_{cls_id}"))
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(conf_val * 100),  # percentage
                    "confidence_score": conf_val,
                    "bbox": [round(coord, 1) for coord in xyxy],
                })

        defect_count = len(detections)
        status = "DEFECTS DETECTED" if defect_count > 0 else "NO DEFECTS DETECTED"
        top_cls = detections[0]["class_id"] if defect_count > 0 else None
        top_conf = detections[0]["confidence"] if defect_count > 0 else 0.0

        return {
            "status": status,
            "defect_count": defect_count,
            "detections": detections,
            "raw_detections": list(detections),
            "processing_time_ms": elapsed_ms,
            "model_name": getattr(model, "model_name", "YOLO11n"),
            "class": top_cls,
            "confidence": top_conf,
        }

    # Fallback PyTorch classification model
    model.eval()
    x = _to_tensor(image)
    with _INFERENCE_LOCK:
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)
            conf, idx = probs.max(dim=1)
    elapsed_ms = (time.time() - t0) * 1000
    cls_id = int(idx.item())
    conf_pct = float(conf.item() * 100)

    detections = [{
        "class_id": cls_id,
        "class_name": DEFECT_CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
        "confidence": conf_pct,
        "confidence_score": float(conf.item()),
        "bbox": [],
    }]

    return {
        "status": "DEFECTS DETECTED",
        "defect_count": 1,
        "detections": detections,
        "raw_detections": list(detections),
        "class": cls_id,
        "confidence": conf_pct,
        "processing_time_ms": elapsed_ms,
        "model_name": "Classifier",
    }


def filter_detections(result: Dict[str, Any], confidence_threshold: float) -> Dict[str, Any]:
    """
    Filter already-predicted detections by confidence threshold in pure Python.
    Avoids expensive, thread-hazardous GPU model reruns when only the threshold filter changes.

    Args:
        result: Prediction result dict from predict()
        confidence_threshold: Cutoff value between 0.0 and 1.0 (e.g. 0.25)

    Returns:
        New prediction result dict with filtered detections and updated summary.
    """
    raw = result.get("raw_detections", result.get("detections", []))
    filtered = []
    for d in raw:
        score = d.get("confidence_score")
        if score is None:
            score = d.get("confidence", 0.0) / 100.0
        if score >= confidence_threshold:
            filtered.append(d)

    defect_count = len(filtered)
    status = "DEFECTS DETECTED" if defect_count > 0 else "NO DEFECTS DETECTED"
    top_cls = filtered[0]["class_id"] if defect_count > 0 else None
    top_conf = filtered[0]["confidence"] if defect_count > 0 else 0.0

    return {
        "status": status,
        "defect_count": defect_count,
        "detections": filtered,
        "raw_detections": raw,
        "processing_time_ms": result.get("processing_time_ms", 0.0),
        "model_name": result.get("model_name", "YOLO11n"),
        "class": top_cls,
        "confidence": top_conf,
    }


def batch_predict(
    images: List[Any],
    model: Any,
    confidence_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Run inference on a batch of images.
    """
    return [predict(img, model, confidence_threshold) for img in images]


def predict_fusion(optical_image: Any, ct_image: Any, model: Any) -> Dict[str, Any]:
    """
    Dual-branch optical+CT fusion model stub for backward compatibility.
    """
    model.eval()
    x_opt = _to_tensor(optical_image)
    x_ct = _to_tensor(ct_image)
    t0 = time.time()
    with torch.no_grad():
        probs = F.softmax(model(x_opt, x_ct), dim=1)
        conf, idx = probs.max(dim=1)
    elapsed_ms = (time.time() - t0) * 1000
    cls_id = int(idx.item())
    return {
        "status": "DEFECTS DETECTED",
        "defect_count": 1,
        "detections": [{
            "class_id": cls_id,
            "class_name": DEFECT_CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            "confidence": float(conf.item() * 100),
            "bbox": [],
        }],
        "class": cls_id,
        "confidence": float(conf.item() * 100),
        "processing_time_ms": elapsed_ms,
        "model_name": "FusionModel",
    }
