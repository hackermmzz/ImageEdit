import torch
from diffusers import QwenImageEditPipeline
import random
import numpy as np
from Tips import *
from Model import AnswerImage
import json
from VLM import AnswerImage
# 配置4bit量化参数
QUANT_CONFIG = {
            "load_in_4bit": True,
            "load_in_8bit": False,
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
    prompt = EDIT_SYSTEM_PROMPT.format(prompt)
    success=False
    while not success:
        try:
            result = AnswerImage([img],prompt)
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
def ImageEditApi(image,prompt:str,neg_prompt=" "):
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(random.randint(0,np.iinfo(np.int32).max)),
        "true_cfg_scale": 4.0,
        "negative_prompt": neg_prompt,
        "num_inference_steps": 50,
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
def EditImage(image,prompt:str,negative_prompt="",polish_prompt=True):
    if polish_prompt:
        prompt=polish_edit_prompt(image,prompt)
    return ImageEditApi(image,prompt,negative_prompt)
    