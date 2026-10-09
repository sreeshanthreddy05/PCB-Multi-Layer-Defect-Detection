from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

Sample = Tuple[str, int, Tuple[int, int, int, int]]


def find_label_file(labels_dir: Path, image_stem: str):
    """
    Find the label corresponding to an image.

    Priority:
    1. Exact filename match
    2. _600 -> _256 fallback
    """

    exact = labels_dir / f"{image_stem}.txt"

    if exact.exists():
        return exact

    if image_stem.endswith("_600"):
        fallback_stem = image_stem[:-4] + "_256"
        fallback = labels_dir / f"{fallback_stem}.txt"

        if fallback.exists():
            return fallback

    return None


def load_dataset(
    data_path: str,
    split: str = "train",
) -> List[Sample]:

    data_path = Path(data_path)

    images_dir = data_path / split / "images"
    labels_dir = data_path / split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")

    dataset = []

    for image_path in sorted(images_dir.iterdir()):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = find_label_file(
            labels_dir,
            image_path.stem
        )

        if label_path is None:
            print(f"Warning: missing label for {image_path.name}")
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as e:
            print(f"Warning: cannot read {image_path.name}: {e}")
            continue

        with open(label_path, "r") as f:

            for line_number, line in enumerate(f, start=1):

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    print(
                        f"Warning: invalid label format in "
                        f"{label_path.name}, line {line_number}"
                    )
                    continue

                try:
                    class_id = int(parts[0])

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                except ValueError:
                    print(
                        f"Warning: invalid values in "
                        f"{label_path.name}, line {line_number}"
                    )
                    continue

                if class_id not in CLASS_NAMES:
                    print(
                        f"Warning: invalid class {class_id} "
                        f"in {label_path.name}"
                    )
                    continue

                xmin = int(
                    (x_center - box_width / 2) * width
                )

                ymin = int(
                    (y_center - box_height / 2) * height
                )

                xmax = int(
                    (x_center + box_width / 2) * width
                )

                ymax = int(
                    (y_center + box_height / 2) * height
                )

                xmin = max(0, min(xmin, width - 1))
                ymin = max(0, min(ymin, height - 1))
                xmax = max(0, min(xmax, width - 1))
                ymax = max(0, min(ymax, height - 1))

                if xmax <= xmin or ymax <= ymin:
                    print(
                        f"Warning: invalid bounding box in "
                        f"{label_path.name}, line {line_number}"
                    )
                    continue

                dataset.append(
                    (
                        str(image_path),
                        class_id,
                        (xmin, ymin, xmax, ymax),
                    )
                )

    return dataset


def validate_dataset(dataset: List[Sample]) -> Dict:

    class_counts = {
        name: 0
        for name in CLASS_NAMES.values()
    }

    image_paths = set()
    corrupt_files = []

    for image_path, class_id, bbox in dataset:

        image_paths.add(image_path)

        class_name = CLASS_NAMES[class_id]
        class_counts[class_name] += 1

        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            corrupt_files.append(image_path)

    return {
        "num_instances": len(dataset),
        "num_images": len(image_paths),
        "num_classes": len(CLASS_NAMES),
        "class_counts": class_counts,
        "num_corrupt_files": len(set(corrupt_files)),
    }


class PCBDataset(Dataset):

    def __init__(self, dataset: List[Sample]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image_path, class_id, bbox = self.dataset[index]

        image = Image.open(image_path).convert("RGB")

        return image, class_id, bbox


def get_dataloader(
    dataset: List[Sample],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
):

    pcb_dataset = PCBDataset(dataset)

    return DataLoader(
        pcb_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def get_dataset_statistics(
    dataset: List[Sample],
) -> Dict:

    statistics = validate_dataset(dataset)

    return statistics