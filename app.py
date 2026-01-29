import gradio as gr

# -----------------------------
# Placeholder backend functions
# -----------------------------

def preview_reconstruction(data_folder, use_paganin, delta, beta, distance):
    """
    Placeholder function that mimics a quick reconstruction preview.
    Returns a single 2D slice (middle slice).
    """
    # In the real implementation:
    # - load projection data from data_folder
    # - optionally apply Paganin phase retrieval
    # - run a lightweight reconstruction
    # - return the middle slice as a numpy array
    import numpy as np
    
    # Dummy image for now
    img = np.random.rand(256, 256)
    return img


def run_full_reconstruction(data_folder, use_paganin, delta, beta, distance):
    """
    Placeholder function that mimics a full reconstruction run.
    """
    # In the real implementation:
    # - load data
    # - apply full preprocessing
    # - run full reconstruction
    # - save results to disk
    return f"Reconstruction completed for data in: {data_folder}"


# -----------------------------
# Gradio GUI
# -----------------------------

with gr.Blocks(title="Phase-Contrast Tomography Reconstruction") as polaris_gui:
    gr.Markdown("## Phase-Contrast Tomography Reconstruction (Exciscope)\n"
                "This interface provides a simplified workflow for running a phase-contrast tomographic reconstruction."
                )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Data")
            data_folder = gr.FileExplorer(
                label="Select projection data folder",
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
            distance = gr.Number(
                value=0.1,
                label="Propagation distance (m)"
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
        fn=preview_reconstruction,
        inputs=[data_folder, use_paganin, delta, beta, distance],
        outputs=preview_image
    )

    run_button.click(
        fn=run_full_reconstruction,
        inputs=[data_folder, use_paganin, delta, beta, distance],
        outputs=status_text
    )


if __name__ == "__main__":
    polaris_gui.launch()
