"""
PCB Defect Detection System - Interactive Web Dashboard
Provides visual PCB defect detection, bounding box overlay, inspection metrics,
and export functionality using Ultralytics YOLO11n.
"""

import os
# Disable Streamlit background file watcher to prevent Windows combase.dll access violation crashes (0xc0000005)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import torch

# Fix for Streamlit's module watcher inspecting torch.classes
try:
    torch.classes.__path__ = []
except Exception:
    pass

from pathlib import Path
import io
import time

import numpy as np
from PIL import Image
import streamlit as st

from src.models.model_loader import load_model
from src.inference.predict import predict, filter_detections
from src.visualization.visualize import visualize_prediction, render_prediction_image
from src.utils.config import load_config
import base64


def render_download_button_html(data_bytes: bytes, file_name: str, label: str = "💾 Download Inspection Result") -> None:
    """
    Renders a direct client-side HTML download button.
    This executes 100% in the browser and does NOT trigger a Streamlit Python server rerun,
    completely preventing server process crashes or WebSocket disconnections.
    """
    b64 = base64.b64encode(data_bytes).decode("utf-8")
    custom_css = """
    <style>
    .dl-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: #0e1117;
        color: #ffffff !important;
        border: 1px solid rgba(250, 250, 250, 0.2);
        padding: 0.45rem 1rem;
        font-size: 0.95rem;
        font-weight: 500;
        border-radius: 0.5rem;
        text-decoration: none !important;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s, background-color 0.2s;
    }
    .dl-btn:hover {
        border-color: #ff4b4b;
        color: #ff4b4b !important;
        background-color: #262730;
    }
    </style>
    """
    html_code = f"""
    {custom_css}
    <a href="data:image/png;base64,{b64}" download="{file_name}" class="dl-btn">
        {label}
    </a>
    """
    st.markdown(html_code, unsafe_allow_html=True)


# Set Streamlit page config
st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading detection model...")
def get_cached_model(checkpoint_path: str):
    """Load and cache the YOLO detection model."""
    path = Path(checkpoint_path)
    if not path.exists():
        # Check alternative common locations
        candidates = [
            Path("runs/pcb_defect/baseline/weights/best.pt"),
            Path("runs/pcb_defect/sanity_check/weights/best.pt"),
            Path("models/best.pt"),
            Path("yolo11n.pt"),
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    if not path.exists():
        return None, f"Checkpoint not found at '{checkpoint_path}'. Please run training first."

    try:
        model = load_model(str(path), model_name="yolo11n")
        return model, str(path)
    except Exception as e:
        return None, str(e)


def main():
    st.title("🔍 PCB Defect Detection System")
    st.caption("Visual inspection assistance prototype powered by YOLO11n object detection")

    st.info(
        "**System Scope & Limitation:** This prototype performs visual PCB defect detection "
        "using surface dataset images. It does not perform X-ray imaging or internal multilayer copper inspection. "
        "Detections assist quality inspection and are not certified manufacturing guarantees."
    )

    # Sidebar settings
    st.sidebar.header("⚙️ Configuration")
    config_file = Path("config.yaml")
    config = load_config(str(config_file)) if config_file.exists() else {}

    default_ckpt = config.get("model", {}).get("checkpoint_dir", "runs/pcb_defect/baseline/weights") + "/best.pt"
    ckpt_path = st.sidebar.text_input("Model Checkpoint Path", value=default_ckpt)

    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(config.get("inference", {}).get("confidence_threshold", 0.25)),
        step=0.05,
        help="Minimum confidence score required to retain detected defects."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Supported Defect Classes")
    st.sidebar.markdown(
        "- 0: **mouse_bite**\n"
        "- 1: **spur**\n"
        "- 2: **missing_hole**\n"
        "- 3: **short**\n"
        "- 4: **open_circuit**\n"
        "- 5: **spurious_copper**"
    )

    # Load Model
    model, load_msg = get_cached_model(ckpt_path)

    if model is None:
        st.error(f"⚠️ Model Loading Error: {load_msg}")
        st.markdown(
            "To train the baseline detector, execute:\n"
            "```powershell\n"
            "python scripts/train_yolo.py --epochs 15 --batch-size 16\n"
            "```"
        )
        return

    st.sidebar.success(f"Loaded model: `{load_msg}`")

    # Main area tabs
    tab_inspect, tab_batch, tab_about = st.tabs(["🖼️ Image Inspection", "📁 Sample Gallery", "ℹ️ About & Metrics"])

    with tab_inspect:
        uploaded_file = st.file_uploader(
            "Choose a PCB image file (.jpg, .png, .jpeg)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Cache raw model predictions keyed strictly by image identity (NOT including conf_thresh).
            # We run inference once with a low threshold (0.01) to gather all candidate defect regions.
            img_cache_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
            if "last_img_cache_key" not in st.session_state or st.session_state["last_img_cache_key"] != img_cache_key:
                with st.spinner("Analyzing PCB surface..."):
                    raw_result = predict(image, model, confidence_threshold=0.01)
                    st.session_state["last_img_cache_key"] = img_cache_key
                    st.session_state["cached_raw_result"] = raw_result
            else:
                raw_result = st.session_state["cached_raw_result"]

            # Filter candidate detections instantly by threshold in Python
            # This completely eliminates GPU re-runs and CUDA race crashes when adjusting the slider!
            result = filter_detections(raw_result, confidence_threshold=conf_thresh)
            rendered_bytes = render_prediction_image(image, result)

            col_orig, col_pred = st.columns(2)
            with col_orig:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
                st.caption(f"Dimensions: {image.size[0]} x {image.size[1]} px | File: {uploaded_file.name}")

            with col_pred:
                st.subheader("Inspection Result")
                st.image(rendered_bytes, use_column_width=True)

                # Direct browser-native download (zero Python server rerun)
                render_download_button_html(
                    data_bytes=rendered_bytes,
                    file_name=f"inspected_{uploaded_file.name}",
                    label="💾 Download Inspection Result",
                )

                # Summary Statistics
                st.markdown("---")
                st.subheader("📊 Inspection Report")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)

                m_col1.metric("Inspection Status", result["status"])
                m_col2.metric("Defects Detected", result["defect_count"])
                m_col3.metric("Processing Time", f"{result['processing_time_ms']:.1f} ms")
                m_col4.metric("Confidence Cutoff", f"{conf_thresh * 100:.0f}%")

                if result["defect_count"] > 0:
                    st.markdown("#### Detected Regions Breakdown")
                    det_data = []
                    for i, det in enumerate(result["detections"], 1):
                        det_data.append({
                            "Index": i,
                            "Class": det["class_name"],
                            "Class ID": det["class_id"],
                            "Confidence": f"{det['confidence']:.1f}%",
                            "Bounding Box (x1, y1, x2, y2)": str(det["bbox"]),
                        })
                    st.table(det_data)
                else:
                    st.success("✅ No defects detected at the selected threshold. (Note: Not an industrial quality certificate).")

    with tab_batch:
        st.subheader("Quick Test from Dataset")
        st.write("Select a sample from the test split to test the detector:")
        sample_paths = list(Path("data/raw/pcb-defect-dataset/test/images").glob("*.jpg"))[:8]
        if sample_paths:
            selected_sample = st.selectbox(
                "Choose sample image:",
                options=sample_paths,
                format_func=lambda p: p.name
            )
            if selected_sample:
                sample_img = Image.open(selected_sample).convert("RGB")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(sample_img, caption="Test Image", use_column_width=True)
                with c2:
                    sample_cache_key = f"sample_{selected_sample.name}"
                    if "last_sample_key" not in st.session_state or st.session_state["last_sample_key"] != sample_cache_key:
                        with st.spinner("Analyzing sample..."):
                            sample_raw = predict(sample_img, model, confidence_threshold=0.01)
                            st.session_state["last_sample_key"] = sample_cache_key
                            st.session_state["cached_sample_raw"] = sample_raw
                    else:
                        sample_raw = st.session_state["cached_sample_raw"]

                    sample_res = filter_detections(sample_raw, confidence_threshold=conf_thresh)
                    sample_img_bytes = render_prediction_image(sample_img, sample_res)
                    st.image(sample_img_bytes, use_column_width=True)
                    st.write(f"**Status:** {sample_res['status']} | **Count:** {sample_res['defect_count']} | **Inference:** {sample_res['processing_time_ms']:.1f} ms")
                    render_download_button_html(
                        data_bytes=sample_img_bytes,
                        file_name=f"inspected_{selected_sample.name}",
                        label="💾 Download Test Inspection",
                    )
        else:
            st.info("Test split images not found.")

    with tab_about:
        st.subheader("System Architecture & Specifications")
        st.markdown(
            """
            - **Detection Architecture:** Ultralytics YOLO11n (2.59M parameters)
            - **Input Resolution:** 640 x 640 px
            - **Inference Runtime:** PyTorch with CUDA acceleration (NVIDIA RTX 4050 Laptop GPU)
            - **Supported Classes:**
                1. `mouse_bite` (edge notch defect)
                2. `spur` (trace copper extension)
                3. `missing_hole` (missing drill hole)
                4. `short` (undesired conductor connection)
                5. `open_circuit` (broken conductor line)
                6. `spurious_copper` (isolated stray copper island)
            """
        )


if __name__ == "__main__":
    main()
