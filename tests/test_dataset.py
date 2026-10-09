"""
Tests for dataset configuration, paths, and YOLO label resolution.
"""

from pathlib import Path
import pytest
import yaml

from src.data.dataset import load_dataset, find_label_file, CLASS_NAMES


def test_dataset_yaml_structure_and_classes():
    """Verify data/pcb_defect.yaml exists and maps all 6 defect classes correctly."""
    yaml_path = Path("data/pcb_defect.yaml")
    assert yaml_path.exists(), "Dataset YAML does not exist"

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert "train" in cfg
    assert "val" in cfg
    assert "test" in cfg
    assert "names" in cfg

    expected_classes = {
        0: "mouse_bite",
        1: "spur",
        2: "missing_hole",
        3: "short",
        4: "open_circuit",
        5: "spurious_copper",
    }
    assert cfg["names"] == expected_classes


def test_label_fallback_and_matching():
    """Verify fallback matching behavior for _600 to _256 stems."""
    labels_dir = Path("data/raw/pcb-defect-dataset/train/labels")
    if not labels_dir.exists():
        pytest.skip("Dataset path not found")

    # Sample known _600 image stem that links to _256
    stem = "light_06_spurious_copper_10_5_600"
    match = find_label_file(labels_dir, stem)
    assert match is not None
    assert match.exists()


def test_split_image_directories_exist():
    """Verify train, val, and test image directories exist."""
    base = Path("data/raw/pcb-defect-dataset")
    for split in ["train", "val", "test"]:
        img_dir = base / split / "images"
        assert img_dir.exists(), f"Missing image directory for {split}"
        assert len(list(img_dir.glob("*.jpg"))) > 0, f"No images found for {split}"
