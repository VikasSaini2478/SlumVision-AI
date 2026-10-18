"""
SlumNet Streamlit dashboard.

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
import json

import numpy as np
import streamlit as st
from PIL import Image
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.common import load_config, get_device, load_checkpoint, overlay_mask
from models.slumnet import build_model
from predict import run_inference
from utils.gis_utils import compute_area_statistics, save_geojson
from utils.xai import GradCAM, overlay_heatmap

CFG_PATH = "configs/config.yaml"
CKPT_PATH = "checkpoints/best_model.pth"


@st.cache_resource
def load_model_cached():
    cfg = load_config(CFG_PATH)
    device = get_device(cfg)
    model = build_model(cfg).to(device)
    if os.path.exists(CKPT_PATH):
        load_checkpoint(CKPT_PATH, model, map_location=device)
        loaded = True
    else:
        loaded = False
    model.eval()
    return model, cfg, device, loaded


def main():
    cfg_preview = load_config(CFG_PATH)
    st.set_page_config(page_title=cfg_preview["dashboard"]["title"], layout="wide")
    st.title(cfg_preview["dashboard"]["title"])
    st.caption("Upload a satellite image chip to detect and map informal settlements.")

    model, cfg, device, ckpt_loaded = load_model_cached()
    if not ckpt_loaded:
        st.warning(
            "No trained checkpoint found at `checkpoints/best_model.pth` — running with "
            "randomly-initialized weights. Train the model first (`python train.py`) for "
            "meaningful predictions."
        )

    with st.sidebar:
        st.header("Settings")
        pixel_area_m2 = st.number_input(
            "Pixel ground area (m²) — leave 0 if unknown", min_value=0.0, value=0.0, step=0.1,
            help="If you know the source imagery's ground sample distance (e.g. Sentinel-2 = 10m -> 100 m²/px), "
                 "enter it here to get real-world area estimates.",
        )
        show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)
        show_expert_usage = st.checkbox("Show Mixture-of-Experts activation", value=True)

    uploaded = st.file_uploader("Upload satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(image)

    with st.spinner("Running inference..."):
        pred_mask, confidence, input_tensor = run_inference(model, image_rgb, cfg, device)
        overlay = overlay_mask(image_rgb, pred_mask, color=(255, 0, 0))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image_rgb, use_column_width=True)
    with col2:
        st.subheader("Segmentation Mask")
        mask_display = (pred_mask * 255).astype(np.uint8) if cfg["data"]["num_classes"] == 2 else pred_mask
        st.image(mask_display, use_column_width=True, clamp=True)
    with col3:
        st.subheader("Overlay")
        st.image(overlay, use_column_width=True)

    st.subheader("Confidence")
    conf_display = (confidence * 255).astype(np.uint8)
    st.image(conf_display, use_column_width=False, width=300, clamp=True)
    st.metric("Mean confidence", f"{np.mean(confidence):.3f}")

    st.subheader("Area & Cluster Statistics")
    stats = compute_area_statistics(
        pred_mask,
        pixel_area_m2=pixel_area_m2 if pixel_area_m2 > 0 else None,
        min_cluster_pixels=cfg["gis"]["min_cluster_pixels"],
    )
    c1, c2 = st.columns(2)
    c1.metric("Detected clusters", stats["num_clusters"])
    c1.metric("Total slum area (px)", stats["total_slum_area_px"])
    if "total_slum_area_hectares" in stats:
        c2.metric("Total slum area (hectares)", f"{stats['total_slum_area_hectares']:.2f}")
    with st.expander("Full statistics JSON"):
        st.json(stats)

    if show_gradcam:
        st.subheader("Explainability — Grad-CAM")
        try:
            layer = dict(model.backbone.named_modules()).get("model.layer4")
            if layer is None:
                layer = list(model.backbone.model.children())[-1]
            cam = GradCAM(model, layer)
            target_class = 1 if cfg["data"]["num_classes"] == 2 else 1
            heatmap = cam(input_tensor, target_class=target_class)
            heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
            cam_overlay = overlay_heatmap(image_rgb, heatmap_resized)
            st.image(cam_overlay, use_column_width=True)
        except Exception as e:
            st.info(f"Grad-CAM unavailable for this backbone/layer combination: {e}")

    if show_expert_usage:
        usage = model.get_expert_usage()
        if usage:
            st.subheader("Mixture-of-Experts Activation (per pyramid level)")
            for i, level_usage in enumerate(usage):
                st.write(f"**Level {i}**")
                st.bar_chart(level_usage)

    st.subheader("Export")
    ec1, ec2 = st.columns(2)
    with ec1:
        mask_bytes = cv2.imencode(".png", mask_display)[1].tobytes()
        st.download_button("Download mask (PNG)", mask_bytes, file_name="slum_mask.png", mime="image/png")
    with ec2:
        report_bytes = json.dumps(stats, indent=2).encode("utf-8")
        st.download_button("Download report (JSON)", report_bytes, file_name="slum_report.json", mime="application/json")


if __name__ == "__main__":
    main()
