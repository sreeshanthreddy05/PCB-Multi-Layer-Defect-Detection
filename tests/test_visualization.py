"""
Tests for src/visualization/visualize.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.visualization.visualize import visualize_prediction


def test_visualize_prediction_produces_output_figure():
    """Verify visualize_prediction() returns a valid matplotlib Figure and renders correctly."""
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    sample_result = {
        "status": "DEFECTS DETECTED",
        "defect_count": 2,
        "detections": [
            {
                "class_id": 0,
                "class_name": "mouse_bite",
                "confidence": 89.5,
                "bbox": [10, 10, 50, 50],
            },
            {
                "class_id": 3,
                "class_name": "short",
                "confidence": 94.2,
                "bbox": [60, 60, 120, 110],
            },
        ],
    }

    fig = visualize_prediction(dummy_img, sample_result)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)


def test_visualize_prediction_saves_file(tmp_path):
    """Verify visualize_prediction() correctly saves to a file when requested."""
    dummy_img = Image.new("RGB", (100, 100), color="blue")
    sample_result = {
        "status": "NO DEFECTS DETECTED",
        "defect_count": 0,
        "detections": [],
    }
    save_file = tmp_path / "test_out.png"
    fig = visualize_prediction(dummy_img, sample_result, save_path=save_file)
    assert save_file.exists()
    assert save_file.stat().st_size > 0
    plt.close(fig)
