from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import torch
from VLM import *
from diffusers import AutoPipelineForInpainting 
import torch
import random
import cv2
from GroundedSam2 import *
from diffusers.utils import load_image, make_image_grid
####################################
#pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
####################################
def Inpainting(image:Image.Image,mask:Image.Image,prompt:str,negative_prompt_list=None):
    negative_prompt=""
    if negative_prompt_list!=None:
        for x in negative_prompt_list:
            negative_prompt=negative_prompt+"."+x
    res = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=8.0,
        true_cfg_scale=1.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=torch.Generator(device="cuda").manual_seed(random.randint(0,np.iinfo(np.int32).max)),
    ).images[0]
    res=res.resize(image.size)
    return res.convert("RGB")
######################################generate mask
def GenerateMask(image: Image.Image,boxes) -> Image.Image:
    width, height = image.size
    new_img = Image.new('RGB', (width, height), color='black')
    new_pixels = new_img.load()
    for box in boxes:
        for y in range(box[1],box[3]):
            for x in range(box[0],box[2]):
                new_pixels[x, y] = (255, 255, 255)  # 纯白
    return new_img.convert("L")

######################################
if __name__=="__main__":
    while True:
        try:
            path=input("image_path:")
            image=Image.open(path).convert("RGB")
            prompt=input("prompt:")
            neg_prompt=input("neg_prompt:")
            res=GroundingDINO_SAM2(image,prompt)
            mask,cutout=res["white_mask"],res["cutOut_img"]
            #局部补全
            Debug("正在进行inpainting...")
            cutout.save("output.jpg")
            mask.save("output.bmp")
            res=Inpainting(image,mask,f"remove {prompt}",[neg_prompt])
            res.save("output.png")
        except Exception as e:
            print("Error:",e)