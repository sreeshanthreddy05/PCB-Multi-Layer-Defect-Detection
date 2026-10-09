"""
Evaluation script for YOLO PCB Defect Detector on validation and held-out test splits.
Generates comprehensive JSON and CSV reports in outputs/reports/.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ultralytics import YOLO

from src.evaluation.metrics import evaluate_yolo_detector



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector on PCB Defect Dataset")
    parser.add_argument("--weights", type=str, default="runs/pcb_defect/baseline/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--data", type=str, default="data/pcb_defect.yaml", help="Dataset YAML")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "both"], help="Split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="outputs/reports", help="Directory for reports")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    weights_path = repo_root / args.weights

    # Fallback if baseline doesn't exist yet, check sanity_check
    if not weights_path.exists():
        fallback_path = repo_root / "runs/pcb_defect/sanity_check/weights/best.pt"
        if fallback_path.exists():
            print(f"Warning: {weights_path} not found. Using fallback {fallback_path}")
            weights_path = fallback_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_to_eval = ["val", "test"] if args.split == "both" else [args.split]
    summary_results = {}

    for split in splits_to_eval:
        print(f"\n==========================================")
        print(f"Evaluating {weights_path.name} on split: {split.upper()}")
        print(f"==========================================")
        metrics = evaluate_yolo_detector(
            model=weights_path,
            data_yaml=str(repo_root / args.data),
            split=split,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
        )

        summary_results[split] = metrics

        print(f"\n--- Results on {split.upper()} ---")
        print(f"Precision:        {metrics['precision']:.4f}")
        print(f"Recall:           {metrics['recall']:.4f}")
        print(f"mAP@50:           {metrics['map50']:.4f}")
        print(f"mAP@50-95:        {metrics['map50_95']:.4f}")
        print(f"Inference latency:{metrics['inference_latency_ms']:.2f} ms")
        print("\nPer-class breakdown:")
        for cname, cdata in metrics["per_class"].items():
            print(f"  {cname:<16}: P={cdata['precision']:.3f}, R={cdata['recall']:.3f}, mAP50={cdata.get('map50', 0.0):.3f}")

        # Save JSON report
        json_path = out_dir / f"evaluation_{split}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved JSON report to {json_path}")

        # Save CSV summary
        csv_path = out_dir / f"evaluation_{split}_per_class.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Class", "Precision", "Recall", "mAP@50"])
            for cname, cdata in metrics["per_class"].items():
                writer.writerow([cname, cdata["precision"], cdata["recall"], cdata.get("map50", "")])
            writer.writerow(["ALL_CLASSES", metrics["precision"], metrics["recall"], metrics["map50"]])
        print(f"Saved CSV per-class report to {csv_path}")

    # Overall summary file
    with open(out_dir / "latest_evaluation_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
