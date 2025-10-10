from diffusers import StableDiffusionInpaintPipeline
import torch
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import threading
#############################################
InpaintingPipe = None
InpaintingPipeLock=threading.Lock()
#############################################
def LoadInpaintingModel():
    global InpaintingPipe
    InpaintingPipe=StableDiffusionInpaintPipeline.from_pretrained(
            "Safetensors/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
    InpaintingPipe.to("cuda")
    InpaintingPipe.image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            "Safetensors/CLIP-ViT-H-14-laion2B-s32B-b79K",  # ✅ 用你本地路径
            local_files_only=True,                 # 强制只用本地文件，避免联网干扰
            torch_dtype=torch.float16
        ).to("cuda")
    InpaintingPipe.load_ip_adapter(
        pretrained_model_name_or_path_or_dict="Safetensors/ip_adapter_plus_sd15", 
        weight_name="ip-adapter-plus_sd15.safetensors",
        subfolder=""
    )
    InpaintingPipe.set_ip_adapter_scale(0.7)
################################################
def Inpainting(image:Image.Image,mask:Image.Image,prompt:str,negative_prompt_list:list=None):
    #
    negative_prompt=""
    if negative_prompt_list!=None:
        for x in negative_prompt_list:
            negative_prompt+=x+' '
    else:
        negative_prompt=None
    #
    InpaintingPipeLock.acquire()
    if InpaintingPipe==None:
        LoadInpaintingModel()
    res = InpaintingPipe(prompt=prompt, image=image, mask_image=mask,height=1024,width=1024,negative_prompt=negative_prompt).images[0]
    InpaintingPipeLock.release()
    return res.resize(image.size)

def InpaintingByIpAdapter(image:Image.Image,mask:Image.Image,prompt:str,adapter_img:Image.Image,negative_prompt_list:list=None):
    #
    negative_prompt=""
    if negative_prompt_list!=None:
        for x in negative_prompt_list:
            negative_prompt+=x+' '
    else:
        negative_prompt=None
    #
    InpaintingPipeLock.acquire()
    if InpaintingPipe==None:
        LoadInpaintingModel()
    res = InpaintingPipe(prompt=prompt, image=image, mask_image=mask,height=1024,width=1024,ip_adapter_image=adapter_img,negative_prompt=negative_prompt).images[0]
    InpaintingPipeLock.release()
    return res.resize(image.size)
#######################
if __name__=="__main__":
    img=Image.open(input("请输入图片:")).convert("RGB")
    mask=Image.open(input("请输入mask:")).convert("L")
    des=input("请输入提示词:")
    adapter_img=Image.open(input("请输入adapter图片:")).convert("RGB")
    res=InpaintingByIpAdapter(img,mask,des,adapter_img)
    res.save("output.png")
    