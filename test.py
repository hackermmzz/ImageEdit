import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image

pipeline = AutoPipelineForInpainting.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to('cuda')

mask = load_image("output.bmp")
blurred_mask = pipeline.mask_processor.blur(mask, blur_factor=33)
blurred_mask.save("output.png")