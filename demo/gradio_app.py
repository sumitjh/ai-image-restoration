import sys
import os
import time
import numpy as np

# Ensure repo root is on sys.path when running as demo/gradio_app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from app.models import enhance_sr, enhance_face, denoise
from app.utils import compute_psnr


def run_sr(image: np.ndarray, scale: int):
    if image is None:
        return None, "Upload an image to get started."
    start = time.time()
    enhanced = enhance_sr(image, scale=int(scale))
    elapsed = int((time.time() - start) * 1000)
    psnr = compute_psnr(image, enhanced, scale=int(scale))
    info = f"Scale: {scale}×  |  PSNR vs bicubic: {psnr:.2f} dB  |  Time: {elapsed} ms"
    return enhanced, info


def run_face(image: np.ndarray):
    if image is None:
        return None, "Upload an image to get started."
    start = time.time()
    enhanced = enhance_face(image)
    elapsed = int((time.time() - start) * 1000)
    return enhanced, f"Processing time: {elapsed} ms"


def run_denoise(image: np.ndarray, strength: int):
    if image is None:
        return None, "Upload an image to get started."
    start = time.time()
    denoised = denoise(image, strength=int(strength))
    elapsed = int((time.time() - start) * 1000)
    return denoised, f"Strength: {strength}  |  Time: {elapsed} ms"


css = ".metrics textarea { font-family: monospace; font-size: 13px; }"

with gr.Blocks(title="AI Image Restoration", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AI Image Restoration
        Super resolution, face restoration, and denoising powered by Real-ESRGAN, GFPGAN, and OpenCV.

        > **First run per model downloads weights (~10–30 s). Subsequent runs are fast.**
        """
    )

    with gr.Tabs():

        # ── Super Resolution ──────────────────────────────────────────────────
        with gr.Tab("Super Resolution"):
            gr.Markdown(
                "Upscale images 2× or 4× with Real-ESRGAN. "
                "PSNR is measured against bicubic upscaling as a baseline — higher is better."
            )
            with gr.Row():
                sr_input  = gr.Image(label="Input",           type="numpy")
                sr_output = gr.Image(label="Enhanced Output", type="numpy")
            with gr.Row():
                sr_scale = gr.Radio(choices=[2, 4], value=4, label="Scale Factor")
                sr_btn   = gr.Button("Enhance →", variant="primary")
            sr_info = gr.Textbox(label="Metrics", interactive=False, elem_classes="metrics")
            sr_btn.click(fn=run_sr, inputs=[sr_input, sr_scale], outputs=[sr_output, sr_info])

        # ── Face Restoration ──────────────────────────────────────────────────
        with gr.Tab("Face Restoration"):
            gr.Markdown(
                "Restore and enhance faces with GFPGAN v1.3. "
                "Works best on portraits with one or more visible faces."
            )
            with gr.Row():
                face_input  = gr.Image(label="Input",            type="numpy")
                face_output = gr.Image(label="Restored Output",  type="numpy")
            face_btn  = gr.Button("Restore →", variant="primary")
            face_info = gr.Textbox(label="Metrics", interactive=False, elem_classes="metrics")
            face_btn.click(fn=run_face, inputs=[face_input], outputs=[face_output, face_info])

        # ── Denoising ─────────────────────────────────────────────────────────
        with gr.Tab("Denoising"):
            gr.Markdown(
                "Remove noise with OpenCV Non-Local Means. "
                "Strength guide: 1–5 light, 10–15 balanced, 20–30 heavy smoothing."
            )
            with gr.Row():
                dn_input  = gr.Image(label="Input",            type="numpy")
                dn_output = gr.Image(label="Denoised Output",  type="numpy")
            dn_strength = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="Strength")
            dn_btn  = gr.Button("Denoise →", variant="primary")
            dn_info = gr.Textbox(label="Metrics", interactive=False, elem_classes="metrics")
            dn_btn.click(fn=run_denoise, inputs=[dn_input, dn_strength], outputs=[dn_output, dn_info])

    gr.Markdown(
        "---\n"
        "Built with "
        "[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) · "
        "[GFPGAN](https://github.com/TencentARC/GFPGAN) · "
        "[OpenCV](https://opencv.org)"
    )


if __name__ == "__main__":
    demo.launch()
