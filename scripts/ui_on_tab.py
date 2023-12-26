import os

import gradio as gr
import modules.scripts as scripts
import numpy as np
from gradio.components.image_editor import Brush
from modules import script_callbacks


def set_image(image):
    # the input here is a single image, the data belongs to the instance you passed in
    layers = image["layers"]
    composite = image["composite"]
    background = image["background"]

    # the value you return will be passed to the instance that is set as the output
    return {
    "layers": layers,
    "background": background
    }

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            img_in = gr.ImageMask(type="pil", brush=Brush(colors=["#000000"], color_mode="defaults"))
            gr.Textbox(lines=5, label="Style Description"),
            gr.CheckboxGroup(["Use Stroke", "Use Example Style", "Use Description"], label="Sampling Options",),
            gr.Slider(minimum=1, maximum=5, step=1, label="Number of Samples"),
            with gr.Row():
                gr.ClearButton()
                gr.Button("Submit", variant="primary")
        with gr.Column():
            img_out = gr.Image(type="pil", label="Output Image", interactive=False)
            
def on_ui_tabs():
        return [(demo, "Unicolor", "unicolor_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
