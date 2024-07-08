import os
import cv2 as cv
import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionPipeline
from groundingdino.util.inference import load_model, load_image, predict, annotate
import requests


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

# Ensure the necessary model and configuration files are downloaded
config_url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
config_path = "./GroundingDINO_SwinT_OGC.py"
os.makedirs(os.path.dirname(config_path), exist_ok=True)

if not os.path.exists(config_path):
    response = requests.get(config_url)
    with open(config_path, "wb") as f:
        f.write(response.content)

model_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
model_path = "./groundingdino_swint_ogc.pth"

if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
        
# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Load GroundingDINO model
model = load_model(config_path, model_path)


def generate_image(text_prompt):
    image = pipe(text_prompt).images[0]
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(1.5)
    return enhanced_image


def main():
    # Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("## GroundingDINO and Stable Diffusion Integration")

        with gr.Tab("Generate Image"):
            prompt = gr.Textbox(label="Text Prompt")
            gen_btn = gr.Button("Generate Image")
            gen_img = gr.Image()

            gen_btn.click(fn=generate_image, inputs=prompt, outputs=gen_img)

    demo.launch()
    

if __name__ == "__main__":
    main()
