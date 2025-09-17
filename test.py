import torch
from diffusers.utils import load_image

# pip install git+https://github.com/huggingface/diffusers
from diffusers import QwenImageControlNetModel, QwenImageControlNetInpaintPipeline

base_model = "Qwen/Qwen-Image"
controlnet_model = "InstantX/Qwen-Image-ControlNet-Inpainting"

controlnet = QwenImageControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)

pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

image = load_image("https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/images/image1.png")
mask_image = load_image("https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/masks/mask1.png")
prompt = "一辆绿色的出租车行驶在路上"

image = pipe(
    prompt=prompt,
    negative_prompt=" ",
    control_image=image,
    control_mask=mask_image,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=control_image.size[0],
    height=control_image.size[1],
    num_inference_steps=30,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
image.save(f"qwenimage_cn_inpaint_result.png")
