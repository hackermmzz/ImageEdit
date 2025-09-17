from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import torch
from PIL import Image
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Krea-dev",
     device_map="cuda",
     torch_dtype=torch.float16  # 可改为 bfloat16 适配 SD3.5 等
)
while True:
    path=input("path:")
    image = Image.open(path).convert("RGB")
    prompt=input("prompt:")
    edited_image = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=6.0,
        num_inference_steps=50
    )
    print(len(edited_image.images))
    edited_image=edited_image.images[0]
    edited_image.save("output.png")