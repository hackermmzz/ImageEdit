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
# 配置4bit量化参数
QUANT_CONFIG = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",       # NV推荐的NF4量化，适配A800
            "bnb_4bit_compute_dtype": torch.bfloat16,  # 计算用bfloat16，A800原生支持
            "bnb_4bit_use_double_quant": True,  # 双量化优化，减少参数冗余
        }
# 使用4bit量化加载模型
pipeline=None

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    device_map="cuda",  # 自动分配设备
    torch_dtype=torch.bfloat16,
    **QUANT_CONFIG
)
pipeline.set_progress_bar_config(disable=None)
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
    w,h=image.size
    client = Ark( 
        base_url="https://ark.cn-beijing.volces.com/api/v3", 
        api_key="723cff33-3b13-420d-ab6d-267800a27475", 
    )
    imagesResponse = client.images.generate( 
        model="doubao-seedream-4-0-250828", 
        prompt=f"{prompt}",
        image=[encode_image(image)],
        size=f"{w}x{h}",
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
    return image.convert("RGB").resize(image.size)    
###################################图像编辑
def ImageEditByPipe(image,prompt:str,neg_prompt:str, prompt_mask=None):
    #
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(random.randint(0,np.iinfo(np.int32).max)),
        "true_cfg_scale": 4,
        "negative_prompt": neg_prompt,
        "num_inference_steps": 50,
        "guidance_scale":6,
        "prompt_embeds_mask":prompt_mask
    }
    with torch.inference_mode():
        output = pipeline(** inputs)
        output_image = output.images[0]
    return output_image.convert("RGB")
###############################给定指令进行编辑
def EditImage(image,prompt:str,negative_prompt_list=None,prompt_mask=None):
    negative_prompt=" "
    if negative_prompt_list:
        for x in negative_prompt_list:
            negative_prompt=negative_prompt+x+"."
    if prompt_mask!=None:
        # 转为 NumPy 数组，并标准化到 0~1
        prompt_mask=prompt_mask.convert("L")
        mask_np = np.array(prompt_mask)
        mask_np = np.where(mask_np>0, 255, 0).astype(np.uint8)
        mask_np = mask_np.astype(np.float32) / 255.0
        # 转换为 torch.Tensor (1, 1, H, W) —— 模型要求的 batch size
        prompt_mask = torch.tensor(mask_np).unsqueeze(0).unsqueeze(0).to("cuda")
    try:
        res=ImageEditByPipe(image,prompt,negative_prompt,prompt_mask)
        return res
    except Exception as e:
        Debug("EditImage:",e)
        return None




################################测试
if __name__=="__main__":
    while True:
        try:
            path=input("path:")
            image=Image.open(path).convert("RGB")
            mask=input("mask_path:")
            mask=Image.open(mask).convert("RGB") if mask!="" else None
            prompt=input("prompt:")
            neg_prompt=input("neg_prompt:")
            res=EditImage(image,prompt,[neg_prompt],mask)
            res.save(f"debug/{RandomImageFileName()}")
        except Exception as e:
            print("error:",e)