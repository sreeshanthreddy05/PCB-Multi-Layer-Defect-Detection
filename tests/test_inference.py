"""
Tests for src/inference/predict.py
"""

import numpy as np
import pytest
from PIL import Image

from src.inference.predict import predict, DEFECT_CLASS_NAMES


class MockYOLOBox:
    def __init__(self, cls_id, conf, xyxy):
        import torch
        self.cls = torch.tensor([cls_id])
        self.conf = torch.tensor([conf])
        self.xyxy = torch.tensor([xyxy])

    def __len__(self):
        return 1


class MockYOLOResult:
    def __init__(self, boxes=None):
        self.boxes = boxes


class MockYOLOModel:
    def __init__(self, mock_detections=None):
        self.model_name = "MockYOLO11n"
        self.names = DEFECT_CLASS_NAMES
        self.mock_detections = mock_detections or []

    def predict(self, source, conf=0.25, verbose=False):
        if not self.mock_detections:
            return [MockYOLOResult(boxes=None)]
        
        import torch
        cls_list = [d["cls"] for d in self.mock_detections]
        conf_list = [d["conf"] for d in self.mock_detections]
        xyxy_list = [d["xyxy"] for d in self.mock_detections]

        class BoxesContainer:
            def __init__(self):
                self.cls = torch.tensor(cls_list)
                self.conf = torch.tensor(conf_list)
                self.xyxy = torch.tensor(xyxy_list)
            def __len__(self):
                return len(cls_list)

        return [MockYOLOResult(boxes=BoxesContainer())]


def test_predict_returns_valid_prediction_structure():
    """Verify predict() returns structured dictionary with status, defect_count, detections, timing."""
    mock_model = MockYOLOModel([
        {"cls": 0, "conf": 0.88, "xyxy": [10.0, 20.0, 50.0, 60.0]}
    ])
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    result = predict(dummy_img, mock_model, confidence_threshold=0.25)

    assert "status" in result
    assert result["status"] == "DEFECTS DETECTED"
    assert "defect_count" in result
    assert result["defect_count"] == 1
    assert "detections" in result
    assert len(result["detections"]) == 1
    assert result["detections"][0]["class_name"] == "mouse_bite"
    assert "processing_time_ms" in result
    assert result["processing_time_ms"] >= 0.0


def test_predict_empty_detections():
    """Verify predict() gracefully handles empty detections without crashing."""
    mock_model = MockYOLOModel([])
    dummy_img = Image.new("RGB", (64, 64), color="black")

    result = predict(dummy_img, mock_model, confidence_threshold=0.25)

    assert result["status"] == "NO DEFECTS DETECTED"
    assert result["defect_count"] == 0
    assert result["detections"] == []
    assert result["class"] is None
    assert result["confidence"] == 0.0


def test_confidence_values_are_within_valid_range():
    """Verify confidence scores are formatted properly within 0-100%."""
    mock_model = MockYOLOModel([
        {"cls": 3, "conf": 0.952, "xyxy": [5.0, 5.0, 25.0, 30.0]}
    ])
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    result = predict(dummy_img, mock_model)
    det = result["detections"][0]
    assert 0.0 <= det["confidence"] <= 100.0
    assert 0.0 <= det["confidence_score"] <= 1.0
    assert det["class_name"] == "short"


def test_filter_detections_filters_by_threshold_accurately():
    """Verify filter_detections accurately filters candidate detections without touching model."""
    from src.inference.predict import filter_detections

    raw_result = {
        "status": "DEFECTS DETECTED",
        "defect_count": 2,
        "processing_time_ms": 12.5,
        "model_name": "YOLO11n",
        "detections": [
            {"class_id": 0, "class_name": "mouse_bite", "confidence": 85.0, "confidence_score": 0.85, "bbox": [10, 10, 20, 20]},
            {"class_id": 1, "class_name": "spur", "confidence": 40.0, "confidence_score": 0.40, "bbox": [30, 30, 40, 40]},
        ],
        "raw_detections": [
            {"class_id": 0, "class_name": "mouse_bite", "confidence": 85.0, "confidence_score": 0.85, "bbox": [10, 10, 20, 20]},
            {"class_id": 1, "class_name": "spur", "confidence": 40.0, "confidence_score": 0.40, "bbox": [30, 30, 40, 40]},
        ],
    }

    # At threshold 0.30: both detections retained
    filtered_low = filter_detections(raw_result, confidence_threshold=0.30)
    assert filtered_low["defect_count"] == 2
    assert filtered_low["status"] == "DEFECTS DETECTED"

    # At threshold 0.50: only mouse_bite retained
    filtered_mid = filter_detections(raw_result, confidence_threshold=0.50)
    assert filtered_mid["defect_count"] == 1
    assert filtered_mid["detections"][0]["class_name"] == "mouse_bite"

    # At threshold 0.90: no detections retained
    filtered_high = filter_detections(raw_result, confidence_threshold=0.90)
    assert filtered_high["defect_count"] == 0
    assert filtered_high["status"] == "NO DEFECTS DETECTED"
    assert filtered_high["class"] is None
