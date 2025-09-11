import os
from PIL import Image
import torch
from tools.prompt_utils import polish_edit_prompt
from diffusers import QwenImageEditPipeline
import random
import numpy as np
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
print("pipeline loaded with 4bit quantization")

# 不需要再调用to("cuda")，因为device_map="auto"已经处理了设备分配
pipeline.set_progress_bar_config(disable=None)
while True:
    path=input("path:")
    prompt=input("prompt:")
    image = Image.open(path).convert("RGB")
    prompt = polish_edit_prompt(prompt, image)
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(random.randint(0,np.iinfo(np.int32).max)),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }

    with torch.inference_mode():
        output = pipeline(** inputs)
        output_image = output.images[0]
        output_image.save("output_image_edit.png")
        print("image saved at", os.path.abspath("output_image_edit.png"))
