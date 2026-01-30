import numpy as np
import gradio as gr
from pipeline import PolarisPipeline
from pathlib import Path

# -----------------------------
# Placeholder backend functions
# -----------------------------

def polaris_data_loader(data_path):

    path = Path(data_path)

    if path.suffix == ".scanlist":
        data_directory = path.parent

    elif path.is_dir():
        data_directory = path

    else:
        raise ValueError("Invalid path: expected a .scanlist file or a directory")
    
    # temp should be parsed later from the gradio interface
    roi = {'angle': (None,None,10),
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

    return pipeline.reconstructed

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
