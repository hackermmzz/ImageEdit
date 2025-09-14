import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

image = pipe(
    "海绵宝宝",
    num_inference_steps=28,
    guidance_scale=3.5,
    local_files_only=True  
).images[0]
image.save("capybara.png")

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16  )
pipe=pipe.to("cuda")

while True:
    x=input(":")
    prompt=input(":")
    neg_prompt=input(":")
    input_image = load_image(x).convert("RGB")

    image = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=2.5,
        negative_prompt=neg_prompt,
        num_inference_steps=50,
    ).images[0]
    image.save("output.png")