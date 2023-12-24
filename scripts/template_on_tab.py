import os

import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks


def greet(image):
    return image.rotate(45)

inputs = [
    gr.Image(label="Input Image", type="pil", show_download_button=True),
    gr.Image(label="Canvas", interactive=False),
    gr.ColorPicker(label="Stroke color"),
    gr.Image(label="Example Style"),
    gr.Textbox(lines=5, label="Style Description"),
    gr.Radio(["Use Stroke", "Use Example Style", "Use Description"], label="Sampling Options"),
    gr.Slider(minimum=1, maximum=10, step=1, label="Number of Samples"),
]

outputs = [
    gr.Image(label="Output Image", show_share_button=True),
]

demo = gr.Interface(
    fn=greet,
    inputs=inputs,
    outputs=outputs,
)

def on_ui_tabs():
        return [(demo, "Extension Template", "extension_template_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
