"""
Simple UI application for the PCB Multi-Layer Defect Detection prototype.

Owner: Person 3 (Inference, Visualization & Application)

IMPORTANT: This file must NOT contain model-training code. It only
wires together: load model -> preprocess image -> predict -> visualize
-> display result.

Flow:
    1. Upload PCB image
    2. Display original image
    3. Run model
    4. Display predicted defect class
    5. Display confidence
    6. Display highlighted defect region
    7. Display PASS/DEFECTIVE status
    8. Display inference time
"""

from typing import Any
import time

import streamlit as st

from src.models.model_loader import load_model
from src.data.preprocessing import preprocess_image
from src.inference.predict import predict, predict_fusion, _to_tensor
from src.models.detector import build_localizer, localize_defect
from src.visualization.visualize import visualize_prediction


def load_app_model(config: dict) -> Any:
    """
    Load the trained model for use by the UI, based on config
    (checkpoint path, model name, num_classes).

    Args:
        config: Loaded project configuration.

    Returns:
        Loaded model instance.
    """
    mcfg = config["model"]
    return load_model(f"{mcfg['checkpoint_dir']}/best.pt", mcfg["name"], mcfg["num_classes"])


def run_inference_pipeline(uploaded_image: Any, model: Any, config: dict) -> dict:
    """
    Run the full UI-facing pipeline for a single uploaded image:
    preprocess -> predict -> visualize.

    Args:
        uploaded_image: Raw image uploaded by the user via the UI.
        model: Loaded model instance.
        config: Loaded project configuration.

    Returns:
        Dictionary with prediction results and the visualized image,
        ready to be rendered in the UI.
    """
    icfg = config["image"]
    processed = preprocess_image(uploaded_image, (icfg["width"], icfg["height"]))
    result = predict(processed, model)

    method = config.get("localization", {}).get("method", "gradcam")
    localizer = build_localizer(model, method)
    result.update(localize_defect(localizer, _to_tensor(processed), result["class"]))

    fig = visualize_prediction(uploaded_image, result)
    return {"prediction": result, "figure": fig}


def format_report(prediction: dict, class_names: list, model_name: str) -> str:
    """
    Format a prediction into the standard PCB Inspection Result report
    (see spec section 12).
    """
    idx = prediction["class"]
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    status = "PASS" if label.upper() == "PASS" else "DEFECTIVE"

    lines = ["PCB Inspection Result", "", f"Status: {status}"]
    if status == "DEFECTIVE":
        lines.append(f"Defect Type: {label}")
    lines += [
        f"Confidence: {prediction['confidence']:.1f}%",
        "",
        f"Model: {model_name}",
        f"Processing Time: {prediction['processing_time_ms']:.1f} ms",
    ]
    return "\n".join(lines)


def main() -> None:
    """
    Entry point for the UI application (e.g. Streamlit app).

    Should not contain any training logic. Only orchestrates the
    upload -> preprocess -> predict -> visualize -> display flow.
    """
    st.title("PCB Defect Detection")

    from src.utils.config import load_config
    config = load_config("config.yaml")
    class_names = config.get("model", {}).get("class_names", ["PASS", "DEFECT"])

    uploaded = st.file_uploader("Upload PCB image", type=["png", "jpg", "jpeg"])
    ct_uploaded = st.file_uploader("Upload CT scan (optional)", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    import numpy as np
    from PIL import Image
    image = np.array(Image.open(uploaded).convert("RGB"))
    st.image(image, caption="Original image")

    icfg = config["image"]

    if ct_uploaded is not None:
        from src.models.classifier import build_fusion_model
        ct_image = np.array(Image.open(ct_uploaded).convert("L"))
        processed_opt = preprocess_image(image, (icfg["width"], icfg["height"]))
        processed_ct = preprocess_image(ct_image, (icfg["width"], icfg["height"]))
        model = build_fusion_model(config["model"]["num_classes"], config["model"]["name"], pretrained=False)
        prediction = predict_fusion(processed_opt, processed_ct, model)
        fig = visualize_prediction(image, prediction)
    else:
        model = load_app_model(config)
        out = run_inference_pipeline(image, model, config)
        prediction, fig = out["prediction"], out["figure"]

    st.pyplot(fig)
    st.text(format_report(prediction, class_names, config["model"]["name"]))


if __name__ == "__main__":
    main()
