
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

def generate_image(prompt, height, width):
    image = pipe(prompt, height=height, width=width).images[0]
    return image

interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Enter Prompt"),
        gr.Slider(256, 768, value=512, step=64, label="Height"),
        gr.Slider(256, 768, value=512, step=64, label="Width"),
    ],
    outputs="image",
    title="AI Image Generation App",
    description="Generate images using Stable Diffusion"
)

if __name__ == "__main__":
    interface.launch()
