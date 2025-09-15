import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import DiffusionPipeline

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
def Inpainting():
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")


    while True:
        path=input("path:")
        mask=input("mask:")
        prompt=input("prompt:")
        image = Image.open(path).convert("RGB")
        mask_image = Image.open(mask).convert("RGB")
        res = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            guidance_scale=8.0,
            num_inference_steps=20,  # steps between 15 and 30 work well for us
            strength=0.99,  # make sure to use `strength` below 1.0
            generator=torch.Generator(device="cuda").manual_seed(0),
        ).images[0]
        res=res.resize(image.size)
        res.save("output.png")
    
def Edit():
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
    pipe=pipe.to("cuda")

    while True:
        x=input("path:")
        prompt=input("prompt:")
        neg_prompt=input("neg_prompt:")
        input_image = load_image(x).convert("RGB")

        image = pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=2.5,
            negative_prompt=neg_prompt,
            num_inference_steps=50,
        ).images[0]
        image.save("output.png")
################################################
if __name__=="__main__":
    mode=int(input("mode:"))
    function=[Inpainting,Edit]
    fun=function[mode]
    while True:
        try:
            fun()
        except Exception as e:
            pass