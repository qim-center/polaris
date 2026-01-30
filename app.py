import numpy as np
import gradio as gr
from pipeline import PolarisPipeline
from dataloader import PolarisDataReader
from pathlib import Path

# -----------------------------
# Placeholder backend functions
# -----------------------------


def to_uint8_rgb(image):
    """
    Convert an arbitrary numeric numpy array into an 8-bit RGB image.

    Accepts:
        - 2D array (grayscale)
        - 3D array with 3 channels (already RGB-like)

    Returns:
        uint8 RGB image with shape (H, W, 3)
    """

    img = np.asarray(image)

    # Replace NaN / inf safely
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to 0–255
    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    img = (img * 255).astype(np.uint8)

    # Convert to RGB
    if img.ndim == 2:
        rgb = np.stack([img] * 3, axis=-1)

    elif img.ndim == 3 and img.shape[-1] == 3:
        rgb = img

    else:
        raise ValueError(f"Unsupported shape for RGB conversion: {img.shape}")

    return rgb


def polaris_data_loader(data_path):

    path = Path(data_path)

    if path.suffix == ".scanlist":
        data_directory = path.parent

    elif path.is_dir():
        data_directory = path

    else:
        raise ValueError("Invalid path: expected a .scanlist file or a directory")
    
    # temp should be parsed later from the gradio interface
    roi = {'angle': (None,None,5),
        'vertical': -1,
        'horizontal': -1,
        }
    
    data = PolarisDataReader(data_directory, roi).read()

    return data

def load_data(scanlist_path, preview):

    data = polaris_data_loader(scanlist_path)
    
    if preview:
        # Here we get only the middle slice
        data = data.get_slice(vertical='centre')

    return data



def run_reconstruction(scanlist_path, use_paganin, delta, beta, energy, preview):
    """
    Placeholder function that mimics a full reconstruction run.
    """

    data = load_data(scanlist_path, preview=preview)
    print (data.shape)

    # Run pipeline
    pipeline = PolarisPipeline(data, delta, beta, energy)
    pipeline.get_sinogram()
    pipeline.correct_rotation()
    pipeline.ring_correction()
    if use_paganin:
        pipeline.paganin()
    pipeline.reconstruct()

    image = pipeline.reconstructed.array

    # Returning just middle slice for viz
    if image.ndim == 3:
        middle_slice = image[image.shape[0] // 2]
    elif image.ndim == 2:
        middle_slice = image
    else:
        raise ValueError(f"Unsupported number of dimensions: {image.ndim}")

    # Set image preview to RGB
    rgb_image = to_uint8_rgb(middle_slice)

    return rgb_image

def run_preview(*args):
    data = run_reconstruction(*args, preview=True)

    return data

def run_full(*args):
    run_reconstruction(*args, preview=False)

    path = args[0]
    print (f"Full reconstruction from {path}")

# -----------------------------
# Gradio GUI
# -----------------------------

with gr.Blocks(title="Phase-Contrast Tomography Reconstruction") as polaris_gui:
    gr.Markdown("## Phase-Contrast Tomography Reconstruction (Exciscope)\n"
                "This interface provides a simplified workflow for running a phase-contrast tomographic reconstruction."
                )

    preview = gr.Checkbox(
                value=True,
                visible=False,
            )
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Data")
            scanlist_path = gr.FileExplorer(
                label="Select the scanlist file.",
                file_count="single"
            )

            gr.Markdown("### Paganin Phase Retrieval")
            use_paganin = gr.Checkbox(
                value=True,
                label="Apply Paganin phase retrieval"
            )

            delta = gr.Number(
                value=1e-6,
                label="Delta",
                precision=6
            )
            beta = gr.Number(
                value=1e-9,
                label="Beta",
                precision=9
            )
            energy = gr.Number(
                value=10000,
                label="Energy"
            )

            preview_button = gr.Button("Preview middle slice")
            run_button = gr.Button("Run reconstruction")

        with gr.Column(scale=3):
            gr.Markdown("### Preview")
            preview_image = gr.Image(
                label="Middle slice preview",
                type="numpy"
            )

            status_text = gr.Textbox(
                label="Status",
                interactive=False
            )

    # -----------------------------
    # Callbacks
    # -----------------------------

    preview_button.click(
        fn=run_preview,
        inputs=[scanlist_path, use_paganin, delta, beta, energy],
        outputs=preview_image
    )

    run_button.click(
        fn=run_full,
        inputs=[scanlist_path, use_paganin, delta, beta, energy],
        outputs=status_text
    )


if __name__ == "__main__":
    polaris_gui.launch()
