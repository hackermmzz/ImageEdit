import torch
from diffusers import QwenImageEditPipeline
import random
import numpy as np
from Tips import *
from Model import AnswerImage
import json
from VLM import AnswerImage
from PIL import Image
import time

# 配置4bit量化参数
QUANT_CONFIG = {
            "load_in_4bit": True,
            "load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4",       # NV推荐的NF4量化，适配A800
            "bnb_4bit_compute_dtype": torch.bfloat16,  # 计算用bfloat16，A800原生支持
            "bnb_4bit_use_double_quant": True,  # 双量化优化，减少参数冗余
        }
# 使用4bit量化加载模型
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    device_map="cuda",  # 自动分配设备
    torch_dtype=torch.bfloat16,
    **QUANT_CONFIG
)

# 不需要再调用to("cuda")，因为device_map="auto"已经处理了设备分配
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
###################################图像编辑
def ImageEditApi(image,prompt:str,neg_prompt, prompt_mask=None):
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(random.randint(0,np.iinfo(np.int32).max)),
        "true_cfg_scale": 4,
        "negative_prompt": neg_prompt,
        "num_inference_steps": 50,
        "width":image.size[0],
        "height":image.size[1],
        "guidance_scale":6,
        "prompt_embeds_mask":prompt_mask
    }
    with torch.inference_mode():
        output = pipeline(** inputs)
        output_image = output.images[0]
    return output_image.convert("RGB")
    '''
    imagesResponse = client.images.generate(
        model="doubao-seededit-3-0-i2i-250628",
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=encode_image(image),
        seed=random.randint(0,np.iinfo(np.int32).max),
        guidance_scale=8.0,
        size="adaptive",
        watermark=True
    )
    #下载图片
    image_url=imagesResponse.data[0].url
    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    response = requests.get(image_url, headers=headers, timeout=30)
    response.raise_for_status() #检查请求是否成功
    #将二进制数据转换为PIL Image对象
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")
    '''
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
        
    res=ImageEditApi(image,prompt,negative_prompt,prompt_mask)
    return res




################################测试
if __name__=="__main__":
    while True:
        path=input("path:")
        image=Image.open(path).convert("RGB")
        prompt=input("prompt:")
        neg_prompt=input("neg_prompt:")
        res=EditImage(image,prompt,[neg_prompt],None)
        res.save(f"debug/{RandomImageFileName()}")