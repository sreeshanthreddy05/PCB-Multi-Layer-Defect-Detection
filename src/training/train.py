"""
Training pipeline.

Owner: Person 2 (ML Model & Training)
"""

from typing import Any, Callable, Dict, List

import copy
import torch
import torch.nn as nn

from src.models.model_loader import save_model
from src.training.validate import validate_model


def train_model(model: Any, train_loader: Any, val_loader: Any, config: Dict[str, Any]) -> Any:
    """
    Train the model using the given data loaders and configuration.

    Must:
      - Set random seeds for reproducibility.
      - Use config-driven hyperparameters (batch_size, epochs,
        learning_rate, optimizer, scheduler).
      - Track training/validation metrics per epoch.
      - Save the best checkpoint via model_loader.save_model().

    Args:
        model: Model instance returned by build_model().
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Full project configuration (training section used).

    Returns:
        Trained model, along with a history of recorded metrics.
    """
    torch.manual_seed(config["data"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tcfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0)
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(tcfg["epochs"]):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ckpt_dir = config.get("model", {}).get("checkpoint_dir", "models")
            save_model(model, f"{ckpt_dir}/best.pt", metadata={"epoch": epoch, **val_metrics})

    return model, history


def train_one_epoch(model: Any, train_loader: Any, optimizer: Any, criterion: Any, device: str = "cpu") -> Dict[str, float]:
    """
    Run a single training epoch.

    Args:
        model: Model being trained.
        train_loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.

    Returns:
        Dictionary of epoch training metrics (e.g. {"loss": ..., "accuracy": ...}).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def tune_hyperparameters(
    build_model_fn: Callable[[], Any],
    train_loader: Any,
    val_loader: Any,
    base_config: Dict[str, Any],
    param_grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run a small grid search over training hyperparameters.

    Args:
        build_model_fn: Zero-arg callable returning a fresh model instance
            (so each trial starts from the same initialization).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        base_config: Full project configuration; "training" section is
            overridden per trial with values from param_grid.
        param_grid: List of training-hyperparameter override dicts,
            e.g. [{"learning_rate": 0.01}, {"learning_rate": 0.001}].

    Returns:
        Dict with "best_params", "best_val_acc", and "results" (per-trial
        val accuracy).
    """
    results = []
    best_val_acc, best_params = -1.0, None

    for params in param_grid:
        trial_config = copy.deepcopy(base_config)
        trial_config["training"].update(params)

        _, history = train_model(build_model_fn(), train_loader, val_loader, trial_config)
        val_acc = history["val_acc"][-1]
        results.append({"params": params, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc, best_params = val_acc, params

    return {"best_params": best_params, "best_val_acc": best_val_acc, "results": results}


def train_model_fusion(model: Any, train_loader: Any, val_loader: Any, config: Dict[str, Any]) -> Any:
    """
    Train the dual-branch FusionModel (optical + CT). Mirrors train_model()
    but batches yield (optical, ct, label) instead of (image, label).

    Args:
        model: FusionModel instance from build_fusion_model().
        train_loader: DataLoader yielding (optical, ct, label) batches.
        val_loader: DataLoader yielding (optical, ct, label) batches.
        config: Full project configuration (training section used).

    Returns:
        Trained model, and history of recorded metrics.
    """
    torch.manual_seed(config["data"].get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tcfg = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0)
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(tcfg["epochs"]):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for optical, ct, labels in train_loader:
            optical, ct, labels = optical.to(device), ct.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(optical, ct)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_metrics = {"loss": total_loss / total, "accuracy": correct / total}

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for optical, ct, labels in val_loader:
                optical, ct, labels = optical.to(device), ct.to(device), labels.to(device)
                outputs = model(optical, ct)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_metrics = {"loss": val_loss / val_total, "accuracy": val_correct / val_total}

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ckpt_dir = config.get("model", {}).get("checkpoint_dir", "models")
            save_model(model, f"{ckpt_dir}/best_fusion.pt", metadata={"epoch": epoch, **val_metrics})

    return model, history
