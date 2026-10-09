"""
Data augmentation.

Owner: Person 1 (Data & Preprocessing)
"""

from typing import Any, Dict

from torchvision import transforms


def build_augmentation_pipeline(config: Dict[str, Any]) -> Any:
    """
    Build an augmentation pipeline (e.g. torchvision transforms or
    albumentations Compose) driven entirely by config parameters.

    Args:
        config: Augmentation section of the project configuration.

    Returns:
        A callable/composable augmentation pipeline.
    """
    ops = []
    if config.get("horizontal_flip"):
        ops.append(transforms.RandomHorizontalFlip())
    if config.get("vertical_flip"):
        ops.append(transforms.RandomVerticalFlip())
    if config.get("rotation_degrees"):
        ops.append(transforms.RandomRotation(config["rotation_degrees"]))
    if config.get("brightness") or config.get("contrast"):
        ops.append(transforms.ColorJitter(
            brightness=config.get("brightness", 0),
            contrast=config.get("contrast", 0),
        ))
    return transforms.Compose(ops)


def apply_augmentation(image: Any, pipeline: Any) -> Any:
    """
    Apply the augmentation pipeline to a single image.

    Args:
        image: Preprocessed image.
        pipeline: Augmentation pipeline returned by
            build_augmentation_pipeline().

    Returns:
        Augmented image.
    """
    return pipeline(image)
