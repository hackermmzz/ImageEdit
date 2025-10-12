import torch
from diffusers import QwenImageEditPipeline
import random
import numpy as np
from Tips import *
from Model import AnswerImage
import json
from VLM import AnswerImage
from PIL import Image
from volcenginesdkarkruntime import Ark 
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions
import requests
import base64
from io import BytesIO
import threading
#######################################
ImageEditPipe=None
#######################################加载图像编辑模型
def LoadImageEdit():
    global ImageEditPipe
    dir="./Safetensors/QwenImageEdit"
    ImageEditPipe = QwenImageEditPipeline.from_pretrained(
        dir,
        device_map="cuda",  # 自动分配设备
        torch_dtype=torch.bfloat16,
    )
    ImageEditPipe.set_progress_bar_config(disable=None)
###############################优化指令
def polish_edit_prompt(img,prompt):
    success=False
    while not success:
        try:
            result = AnswerImage([img],EDIT_SYSTEM_PROMPT,f"User Input: {prompt}")
            if isinstance(result, str):
                result = result.replace('```json','')
                result = result.replace('```','')
                result = json.loads(result)
            else:
                result = json.loads(result)
            polished_prompt = result['Rewritten']
            polished_prompt = polished_prompt.strip()
            polished_prompt = polished_prompt.replace("\n", " ")
            success = True
        except Exception as e:
            print(f"[Warning] Error during API call: {e}")
    return polished_prompt
###################################图像编辑(API)
def ImageEditByAPI(image,prompt:str,neg_prompt:str)->Image.Image:
    client = Ark( 
        base_url="https://ark.cn-beijing.volces.com/api/v3", 
        api_key="91d86bec-d21e-4c87-b121-bbf249b50345", 
    )
    
    input=image
    w,h=image.size
    image.resize((1024,1024))
    imagesResponse = client.images.generate( 
        model="doubao-seedream-4-0-250828", 
        prompt=f'''{prompt} '''+('''and don't occur following cases: "{neg_prompt}"''' if neg_prompt else ""),
        image=[encode_image(input)],
        size=f"4096x4096",
        sequential_image_generation="auto",
        sequential_image_generation_options=SequentialImageGenerationOptions(max_images=1),
        response_format="url",
        watermark=False
    )
    url=None
    for image in imagesResponse.data:
        url=image.url
        break
    response = requests.get(url,timeout=30)
    response.raise_for_status() #检查请求是否成功
    #将二进制数据转换为PIL Image对象
    image = Image.open(BytesIO(response.content)).convert("RGB")
    #保持生成的图片尺寸和原来一样
    # image=image.resize((w,h))
    return image
###################################图像编辑
def ImageEditByPipe(image:Image.Image,prompt:str,neg_prompt:str):
    global ImageEditPipe
    if ImageEditPipe==None:
        LoadImageEdit()
    #
    size=(1024,1024)
    #
    inputs = {
        "image": [image.resize(size)],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": neg_prompt,
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "height":size[0],
        "width":size[1]
    }
    res=None
    with torch.inference_mode():
        output = ImageEditPipe(**inputs)
        res = output.images[0]
    res=res.convert("RGB")
    return res
###############################给定指令进行编辑
def EditImage(image,prompt:str,negative_prompt_list=None):
    negative_prompt=f"{PreDefine_NegPrompt}"
    if negative_prompt_list:
        for x in negative_prompt_list:
            negative_prompt=x+","+negative_prompt
    try:
        if Enable_Local_ImageEdit:
            return ImageEditByPipe(image,prompt,negative_prompt)
        else:
            return ImageEditByAPI(image,prompt,negative_prompt)
    except Exception as e:
        Debug("EditImage:",e)
        return EditImage(image,prompt,negative_prompt_list)
################################修复图像
def ImageFixByAPI(images,prompt:str)->Image.Image:
    #images[0]是原图,images[1]是编辑过的图
    client = Ark( 
        base_url="https://ark.cn-beijing.volces.com/api/v3", 
        api_key="0768c60e-15da-44c5-9205-2ebf5a1594cf", 
    )
    w,h=images[0].size
    if w*h<921600:
        scale=(921600/(w*h))**(0.5)
        input0=images[0].resize((int(w*scale)+2,int(h*scale)+2))
        input1=images[1].resize((int(w*scale)+2,int(h*scale)+2))
        images=[input0,input1]
    
    imagesResponse = client.images.generate( 
        model="doubao-seedream-4-0-250828", 
        prompt=f'''{prompt}''',
        image=[encode_image(input) for input in images],
        size=f"2048x2048",
        sequential_image_generation="auto",
        sequential_image_generation_options=SequentialImageGenerationOptions(max_images=1),
        response_format="url",
        watermark=False
    )
    url=None
    for image in imagesResponse.data:
        url=image.url
        break
    response = requests.get(url,timeout=30)
    response.raise_for_status() #检查请求是否成功
    #将二进制数据转换为PIL Image对象
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB").resize((w,h))               
#############################################
if __name__=="__main__":
    img=Image.open(input("img:")).convert("RGB")
    prompt=input("prompt:")
    res=ImageEditByAPI(img,prompt,"")
    res.save("debug/origin.png")
    res.resize(img.size).save("debug/resize.png")