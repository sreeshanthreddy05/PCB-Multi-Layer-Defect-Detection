"""
Model checkpoint saving and loading supporting YOLO object detection models
and backwards compatibility for classification models.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import json

import torch


def save_model(model: Any, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a trained model checkpoint along with metadata.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save"):
        # Ultralytics YOLO model
        model.save(str(target))
    elif isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), str(target))
    else:
        torch.save(model, str(target))

    if metadata:
        meta_path = target.with_suffix(target.suffix + ".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_model(path: str, model_name: str = "yolo11n", num_classes: int = 6) -> Any:
    """
    Load a model checkpoint from disk. Supports YOLO checkpoints (.pt)
    as well as PyTorch state_dict models.

    Args:
        path: Path to checkpoint (.pt).
        model_name: Model identifier ('yolo11n', 'yolo', etc.)
        num_classes: Number of defect classes.

    Returns:
        Loaded detector or model.
    """
    chk_path = Path(path)
    if not chk_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {chk_path}")

    # Check if this is a YOLO model
    if "yolo" in model_name.lower() or chk_path.suffix == ".pt":
        try:
            from ultralytics import YOLO
            model = YOLO(str(chk_path))
            try:
                model.fuse()
            except Exception:
                pass
            return model
        except Exception:
            # Fall back to PyTorch state_dict loading if not YOLO
            pass

    from src.models.classifier import build_model
    model = build_model(num_classes, model_name, pretrained=False)
    model.load_state_dict(torch.load(chk_path, map_location="cpu"))
    model.eval()
    return model
