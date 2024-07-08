import cv2 as cv
import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import StableDiffusion3Pipeline
from groundingdino.util.inference import load_model, load_image, predict, annotate
import logging


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'


config_path = "./models/GroundingDINO_SwinT_OGC.py"
model_path = "./models/groundingdino_swint_ogc.pth"

# Load Stable Diffusion model
try:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", device_map="auto", torch_dtype=torch.float16 if device == "cuda" else None).to(device)
except Exception as e:
    logging.error(f"Error loading Stable Diffusion model: {e}")
    logging.debug("Please check the model path and try again.")
    pipe = None

# Load GroundingDINO model
try:
    model = load_model(config_path, model_path)
except Exception as e:
    logging.error(f"Error loading GroundingDINO model: {e}")
    logging.debug("Please check the model path and try again.")
    model = None

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
    try:
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


def main():
    # Gradio Interface
    with gr.Blocks() as demo:
        gr.Markdown("## GroundingDINO and Stable Diffusion Integration")

        with gr.Tab("Generate Image"):
            prompt = gr.Textbox(label="Text Prompt")
            gen_btn = gr.Button("Generate Image")
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


if __name__ == "__main__":
    main()
