import logging
import argparse
from get_models import download_config, download_model

import cv2 as cv
import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline
from groundingdino.util.inference import load_model, load_image, predict, annotate


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'

config_path = "./models/GroundingDINO_SwinT_OGC.py"
model_path = "./models/groundingdino_swint_ogc.pth"

version = None

def predict_image(img, text, box_threshold, text_threshold):
    if img is None:
        raise ValueError("Image is not loaded properly")

    img_array, img_tensor = load_image(img)
    
    boxes, logits, phrases = predict(
        model=model,
        image=img_tensor,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    annotated_frame = annotate(image_source=img_array, boxes=boxes, logits=logits, phrases=phrases)
    return Image.fromarray(cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB))


def generate_image(text_prompt):
    global version
    try:
        if version:
            image = pipe(text_prompt).images[0]
        else:
            image = pipe(text_prompt,
                        negative_prompt="",
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        ).images[0]
    except Exception as e:
        logging.error(f"Error generating image: {e}")
        logging.debug("Please check the text prompt and try again.")
        logging.debug(f"Text {text_prompt}")
        return None

    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(1.5)
    # save the image
    enhanced_image.save("../data/generated_image.png")
    return enhanced_image


def main(args):
    global version
    if args.version1:
        version = True
    else:
        version = False

    # Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("## GroundingDINO and Stable Diffusion Integration")

        with gr.Tab("Generate Image"):
            prompt = gr.Textbox(label="Text Prompt")
            gen_btn = gr.Button("Generate Image")
            save_btn = gr.Button("Save Image")
            gen_img = gr.Image()

            gen_btn.click(fn=generate_image, inputs=prompt, outputs=gen_img)

        with gr.Tab("Predict Image"):
            img_input = gr.Image(type="filepath", label="Input Image")
            caption = gr.Textbox(label="Caption")
            box_thresh = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Box Threshold")
            text_thresh = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.01, label="Text Threshold")
            pred_btn = gr.Button("Predict")
            pred_img = gr.Image()

            pred_btn.click(fn=predict_image, inputs=[img_input, caption, box_thresh, text_thresh], outputs=pred_img)

    demo.launch(inbrowser=True)
    

def get_args():
    parser = argparse.ArgumentParser(description="GroundingDINO and Stable Diffusion Integration")
    parser.add_argument("-v1", "--version1", action="store_true", help="Gets the v1 version of the StableDiffusion model, else it will get the v3 version.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        if args.version1:
            pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
            logging.info(f"Stable Diffusion model version: {pipe}")
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", device_map="auto", torch_dtype=torch.float16 if device == "cuda" else None).to(device)
            logging.info(f"Stable Diffusion model version: {pipe}")

    except Exception as e:
        logging.error(f"Error loading Stable Diffusion model: {e}")
        logging.debug("Please check the model.")
        pipe = None

    # Load GroundingDINO model
    try:
        model = load_model(config_path, model_path)
    except Exception as e:
        logging.error(f"Error loading GroundingDINO model: {e}")
        logging.debug("Checking the model and config path...")
        download_config()
        download_model()
        model = None

    main(args)
