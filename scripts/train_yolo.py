"""
Training entry point for PCB Defect Detection using Ultralytics YOLO11n.
Saves model checkpoint, training metrics, and logs in runs/pcb_defect/baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 on PCB Defect Dataset")
    parser.add_argument("--data", type=str, default="data/pcb_defect.yaml", help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for GPU or 'cpu'")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers (0 on Windows)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--project", type=str, default="runs/pcb_defect", help="Project directory")
    parser.add_argument("--name", type=str, default="baseline", help="Run name")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    data_path = repo_root / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    # Determine device
    if not args.device:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"=== Starting YOLO Training ===")
    print(f"Data YAML: {data_path}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print(f"Patience: {args.patience}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")

    # Load pretrained model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        patience=args.patience,
        device=device,
        workers=args.workers,
        seed=args.seed,
        project=str(repo_root / args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else repo_root / args.project / args.name
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\nTraining finished.")
    print(f"Output directory: {save_dir}")
    print(f"Best checkpoint exists: {best_pt.exists()} ({best_pt})")

    # Record reproducibility info
    meta = {
        "model_name": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "device": str(device),
        "seed": args.seed,
        "save_dir": str(save_dir),
        "best_checkpoint": str(best_pt) if best_pt.exists() else None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    with open(save_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
