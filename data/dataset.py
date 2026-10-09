"""
Dataset loading and splitting.

Owner: Person 1 (Data & Preprocessing)

Responsible for loading the raw PCB defect image dataset, validating it,
and producing reproducible train/validation/test splits.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class PCBDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_dataset(data_path: str) -> Any:
    """
    Load the raw PCB defect dataset from disk.

    Args:
        data_path: Path to the dataset root directory (config-driven,
            not hard-coded).

    Returns:
        A dataset object or list of (image_path, label) pairs.
    """
    root = Path(data_path)
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    samples = []
    for label, class_name in enumerate(class_names):
        for img_path in (root / class_name).glob("*"):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                samples.append((str(img_path), label))
    return samples


def validate_dataset(dataset: Any) -> Dict[str, Any]:
    """
    Validate the dataset: check for corrupt/missing images, verify
    label integrity, and report basic statistics.

    Args:
        dataset: Dataset object returned by load_dataset().

    Returns:
        Dictionary summarizing validation results (e.g. num_samples,
        num_classes, num_corrupt_files).
    """
    num_corrupt = 0
    for path, _ in dataset:
        try:
            Image.open(path).verify()
        except Exception:
            num_corrupt += 1

    labels = {label for _, label in dataset}
    return {
        "num_samples": len(dataset),
        "num_classes": len(labels),
        "num_corrupt_files": num_corrupt,
    }


def split_dataset(
    dataset: Any,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Split the dataset into train/validation/test subsets.

    Args:
        dataset: Full dataset object.
        train_split: Fraction of data used for training.
        val_split: Fraction of data used for validation.
        test_split: Fraction of data used for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_set, val_set, test_set).
    """
    train_val, test = train_test_split(
        dataset, test_size=test_split, random_state=seed,
        stratify=[l for _, l in dataset],
    )
    rel_val = val_split / (train_split + val_split)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed,
        stratify=[l for _, l in train_val],
    )
    return train, val, test


def get_dataloader(dataset: Any, batch_size: int, shuffle: bool = True) -> Any:
    """
    Wrap a dataset split into a framework-specific DataLoader.

    Args:
        dataset: A train/val/test dataset split.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples each epoch.

    Returns:
        A DataLoader instance (e.g. torch.utils.data.DataLoader).
    """
    ds = dataset if isinstance(dataset, Dataset) else PCBDataset(dataset)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


class PairedPCBDataset(Dataset):
    """
    Dataset for multi-modal samples: each item is (optical_path, ct_path, label).
    Returns (optical_tensor, ct_tensor, label). Used by the fusion model
    once CT scans are available; optical-only training continues to use
    PCBDataset unchanged.
    """

    def __init__(self, samples: List[Tuple[str, str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        optical_path, ct_path, label = self.samples[idx]
        optical = Image.open(optical_path).convert("RGB")
        ct = Image.open(ct_path).convert("L")
        if self.transform:
            optical = self.transform(optical)
            ct = self.transform(ct)
        return optical, ct, label
