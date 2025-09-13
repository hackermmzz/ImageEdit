import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

while True:
    x=input(":")
    prompt=input(":")
    neg_prompt=input(":")
    input_image = load_image(x).convert("RGB")

    image = pipe(
        image=input_image,
        prompt=prompt,
        guidance_scale=2.5,
        negative_prompt=neg_prompt,
        num_inference_steps=50
    ).images[0]
    image.save("output.png")