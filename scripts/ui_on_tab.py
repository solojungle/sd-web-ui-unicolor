import os

import gradio as gr
import modules.scripts as scripts
import numpy as np
# from gradio.components.image_editor import Brush
from modules import script_callbacks
from PIL import Image

from scripts.unicolor.colorizer import Colorizer

device = 'mps'
ckpt_file = os.path.join(scripts.basedir(), "scripts/unicolor/framework/checkpoints/unicolor_mscoco/mscoco_step259999.ckpt")


# Load CLIP and ImageWarper for text-based and exemplar-based colorization 
colorizer = Colorizer(ckpt_file, device, [256, 256], load_clip=True, load_warper=True)

def colorize_image(input_image):
    # convert numpy.ndarray to PIL.Image
    gray_image = Image.fromarray(input_image.astype('uint8'), 'RGB').convert('L')

    # Run the input image through the colorizer
    output_image = colorizer.sample(gray_image, [], topk=100)

    # convert PIL.Image to numpy.ndarray
    return np.array(output_image)

with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image()
            gr.Textbox(lines=5, label="Style Description"),
            gr.CheckboxGroup(["Use Stroke", "Use Example Style", "Use Description"], label="Sampling Options",),
            gr.Slider(minimum=1, maximum=5, step=1, label="Number of Samples"),
            with gr.Row():
                gr.ClearButton()
                submit = gr.Button(value="Colorize")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Output Image", interactive=False)
            submit.click(fn=colorize_image, inputs=image_input, outputs=image_output)

def on_ui_tabs():
        return [(demo, "Unicolor", "unicolor_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
