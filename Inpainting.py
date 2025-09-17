from PIL import Image
import numpy as np
import torch
from VLM import *
from diffusers import AutoPipelineForInpainting 
import torch
import random
import cv2
from VLM import GetROE
####################################
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
####################################
def Inpainting(image:Image.Image,mask:Image.Image,prompt:str):
    res = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=8.0,
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
######################################get box
from PIL import ImageDraw
def DrawRedBox(image, boxes, width=3):
    # 拷贝原图避免修改原图像
    image_copy = image.copy()
    # 创建可绘制对象
    draw = ImageDraw.Draw(image_copy)
    # 画红框
    for box in boxes:
        draw.rectangle(box, outline="red", width=width)
    return image_copy
#inpainiting
def InpaintingArea(image:Image,task:str):
    #get box
    boxes=GetROE(image,f"Now I will give you the image-edit instruction:{task}.You should give me the fittable answer as a mask for inpainting")
    Debug("the box is:",boxes)
    #get mask
    mask=GenerateMask(image,boxes)
    #inpainting
    res=Inpainting(image,mask,task)
    return res
######################################
if __name__=="__main__":
    while True:
        try:
            path=input("image_path:")
            image=Image.open(path).convert("RGB")
            prompt=input("prompt:")
            res=InpaintingArea(image,prompt)
            res.save("output.png")
        except Exception as e:
            print("Error:",e)