import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import DiffusionPipeline
from VLM import *
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
        
def alpha_to_white_black_mask(image: Image.Image,box:tuple) -> Image.Image:
    width, height = image.size
    new_img = Image.new('RGB', (width, height), color='black')
    pixels = image.load()
    new_pixels = new_img.load()
    for y in range(box[1],box[3]):
        for x in range(box[0],box[2]):
            new_pixels[x, y] = (255, 255, 255)  # 纯白
    return new_img

def GetObjectBox():
    path=input("path:")
    target=input("target:")
    image=Image.open(path).convert("RGB")
    role_tip='''
        You are now an image object bounding box detection expert. I will provide you with an image and a prompt of the target object to be bounded, and you need to give the answer in accordance with the following rules:
        (1) If there are multiple target objects, return multiple results; if there is only one, return one result; if there are none, return "none".
        (2) For each result, provide a four-tuple (x0, y0, x1, y1), where each element is a floating-point number between 0 and 1, representing the relative position of the target from its top-left corner to bottom-right corner in the image.
        (3) The final result should follow the following format: [(ans0), (ans1), ...]
        Remember: Only need to provide the answer, without any additional responses.
    '''
    question=f"target:{target}"
    res=AnswerImage([image],role_tip,question)
    print(res)
    res=[float(s.strip()) for s in res.strip("[()]").split(",")]
    print(res)
    w,h=image.size
    res=(int(res[0]*w),int(res[1]*h),int(res[2]*w),int(res[3]*h))
    image=alpha_to_white_black_mask(image,res)
    image.save("output.png")
###############################################
if __name__=="__main__":
    mode=int(input("mode:"))
    function=[Inpainting,Edit,GetObjectBox]
    fun=function[mode]
    while True:
        try:
            fun()
        except Exception as e:
            print("Error:"+e)