# PCB Defect Detection System (Visual Prototype)

An end-to-end computer-vision system for automated visual inspection of printed circuit board (PCB) defects using deep learning object detection (Ultralytics YOLO11n).

---

## 1. Project Objectives & Problem Statement

Manual quality inspection of printed circuit boards is error-prone, labor-intensive, and inconsistent at scale. Microscopic copper and substrate flaws—such as mouse bites, spurious copper, or hairline open circuits—can cause intermittent board failure or complete system breakdown.

This project delivers an automated visual inspection assistance prototype that:
1. Accepts optical PCB surface images.
2. Detects, localizes, and classifies defect regions into six standard defect categories.
3. Quantifies detection confidence and overlays precise bounding boxes and labels.
4. Generates structured inspection reports with defect counts and processing latencies.
5. Surfaces inspection results in an interactive web application suitable for laboratory and pitch demonstration.

---

## 2. Critical Prototype Limitations

> **Academic Prototype Notice:**  
> The current prototype performs dataset-based visual PCB defect detection using annotated surface PCB images. It does not perform X-ray imaging or verified internal multilayer inspection. Detections provide quality-control inspection assistance and are not certified manufacturing quality guarantees. Future work may integrate controlled optical imaging hardware and evaluate its suitability for different PCB inspection conditions.

---

## 3. Dataset & Supported Defect Classes

The prototype is trained and evaluated on the standardized YOLO PCB defect dataset containing **10,668 images** and **21,664 annotated defect instances** across train, validation, and test splits.

### Supported Defect Categories (Classes 0–5)
* `0`: **mouse_bite** — Edge notch / substrate void defect
* `1`: **spur** — Unintended copper projection protruding from a trace
* `2`: **missing_hole** — Drilled via or through-hole absent from designated pad
* `3`: **short** — Unintended conductor bridge linking adjacent copper traces
* `4`: **open_circuit** — Discontinuous trace / broken conductor line
* `5`: **spurious_copper** — Stray, isolated copper artifact left after etching

### Split Breakdown
| Split | Images | Defect Instances |
| :--- | :---: | :---: |
| **Train** | 8,534 | 17,342 |
| **Validation** | 1,066 | 2,164 |
| **Test (Held-Out)** | 1,068 | 2,158 |
| **Total** | **10,668** | **21,664** |

---

## 4. Architecture & Pipeline

* **Detector Architecture:** Ultralytics YOLO11n (nano), 101 layers, 2.58M parameters, 6.3 GFLOPs.
* **Input Resolution:** 640 × 640 pixels (RGB).
* **Hardware:** CUDA-accelerated on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
* **Inference Pipeline:** Image ingestion $\rightarrow$ Tensor preprocessing $\rightarrow$ YOLO11n detection $\rightarrow$ Non-maximum suppression $\rightarrow$ Bounding box & label visualization.
* **Separation of Concerns:** Input image acquisition is decoupled from inference, allowing future camera adapters (`CameraInput`) to plug directly into the pipeline.

---

## 5. Measured Evaluation Results

Evaluated on held-out test and validation sets:

### Test Set Performance (1,068 Images, 2,158 Instances)
* **Precision (P):** `0.8320` (83.2%)
* **Recall (R):** `0.7183` (71.8%)
* **mAP@50:** `0.8028` (80.3%)
* **mAP@50-95:** `0.3603` (36.0%)
* **Average Inference Latency:** `~4.75 ms` per image (~210 FPS on RTX 4050)

#### Per-Class Breakdown (Test Set)
| Class ID | Defect Name | Precision | Recall | mAP@50 |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `mouse_bite` | 0.695 | 0.741 | 0.742 |
| 1 | `spur` | 0.883 | 0.615 | 0.760 |
| 2 | `missing_hole` | 0.989 | 0.960 | 0.986 |
| 3 | `short` | 0.794 | 0.620 | 0.749 |
| 4 | `open_circuit` | 0.898 | 0.652 | 0.810 |
| 5 | `spurious_copper` | 0.733 | 0.722 | 0.770 |

---

## 6. Quick Start & Execution Guide

### Prerequisites
* Python 3.10+
* PyTorch & CUDA (optional but recommended for real-time inference)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Tests
```powershell
python -m pytest tests/
```

### 3. Launch the Interactive Dashboard
```powershell
python -m streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run Model Training
```powershell
python scripts/train_yolo.py --epochs 15 --batch-size 16 --patience 5
```

### 5. Run Model Evaluation
```powershell
# Evaluate on held-out test split
python scripts/evaluate_yolo.py --weights runs/pcb_defect/baseline/weights/best.pt --split test

# Evaluate on validation split
python scripts/evaluate_yolo.py --weights runs/pcb_defect/baseline/weights/best.pt --split val
```
Reports and metrics are saved to `outputs/reports/`.
