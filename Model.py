import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForImageTextToText,AutoModelForZeroShotObjectDetection
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from Tips import *
import xml.etree.ElementTree as ET
import json
from dashscope import MultiModalConversation
import requests
import numpy as np
from io import BytesIO
from transformers import pipeline
from openai import OpenAI
import base64
from transformers import CLIPModel, CLIPProcessor
from scipy.spatial.distance import cosine
from volcenginesdkarkruntime import Ark
import random
######################################指定设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#######################################
class Expert():
    def __init__(self,processor=None,model=None,generator=None):
        self.processor=processor
        self.model=model
        self.generator=generator
    def ToDevice(self,device):
        to_list=[self.processor,self.model,self.generator]
        for x in to_list:
            if x:
                x.to(device)
        if self.model:
            self.model.eval()
##############定义各种模型
LLM=None    #使用deepseek-r1作为LLM
VLM=None    #使用GLM4.1作为VLM
Editor=None #图像编辑模型暂时没有
GroundingDINO=None
SAM=None
CLIP=None
######################################
######################################异步加载所有模型
def LoadAllModel():
    def LoadLLM():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 启用 4bit 量化
            bnb_4bit_use_double_quant=True,  # 双量化，进一步减少显存占用
            bnb_4bit_quant_type="nf4",  # 推荐的量化类型（比 fp4 更适合自然语言）
            bnb_4bit_compute_dtype=torch.bfloat16  # 计算时用 bfloat16，保精度
        )
        processor = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            quantization_config=bnb_config, 
            device_map="auto", 
            trust_remote_code=True
        ).to(DEVICE).eval()
        global LLM
        LLM=Expert(processor,model)
    def LoadVLM():
        QUANT_CONFIG = {
            "load_in_4bit": False,
            "load_in_8bit": False,
            "low_cpu_mem_usage": False,
        }
        processor = AutoProcessor.from_pretrained("zai-org/GLM-4.1V-9B-Thinking")
        model = AutoModelForImageTextToText.from_pretrained(
            "zai-org/GLM-4.1V-9B-Thinking",
            trust_remote_code=True,  # 必须开启（多模态模型结构需远程代码）
            device_map="auto",  # 自动分配设备（优先用GPU，剩余放CPU）
            **QUANT_CONFIG  # 启用显存优化（若CPU运行，删除这一行）
            ).to(DEVICE).eval()
        global VLM
        VLM=Expert(processor,model)
    def LoadGroundingDINO():
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(DEVICE)
        global GroundingDINO
        GroundingDINO=Expert(processor,model)  
    def LoadSAM():
        generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=0)
        global SAM
        SAM=Expert(generator=generator)
    def LoadClip():
        model_name = "openai/clip-vit-large-patch14"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        global CLIP
        CLIP=Expert(processor=processor,model=model)
    #多线程加载
    tasks=[
        #    LoadVLM,
        #    LoadLLM,
            LoadGroundingDINO,
            LoadSAM,
            LoadClip,
        ]
    for task in tasks:
        task()
    Debug("所有模型加载完毕")
######################################
def ExtractAnswer(data:str):
        think=""
        answer=""
        beg0=data.find("<answer>")
        end0=data.find("</answer>") 
        if beg0!=-1 and end0!=-1:
            answer=data[beg0+len("<answer>"):end0]
        beg1=data.find("<think>")
        end1=data.find("</think>")
        if beg1!=-1 and end1!=-1:
            think=data[beg1+len("<think>"):end1]
        return think,answer
#######################################编码图片
def encode_image(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:{'image/jpeg'};base64,{encoded_string}"
#######################################
client = Ark(
    # 此为默认路径，您可根据业务所在地域进行配置
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
    api_key="723cff33-3b13-420d-ab6d-267800a27475",
    timeout=1800,
    # 设置重试次数
    max_retries=2,
)
def AnswerText(question:str):
    ##########################调用豆包大模型
    try:
        # Non-streaming:
        completion = client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model="doubao-1-5-pro-256k-250115",
            messages=[
                {"role": "user", "content": f"{question}"},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        Debug(e)
        return AnswerText(question)
    ############################
        processor=LLM.processor
        model=LLM.model
        # 4. 定义对话内容
        messages = [
            {"role": "user", "content":"{}".format(question)},
        ]

        # 5. 处理输入（转为模型可识别的格式，并移到模型所在设备）
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to(DEVICE)  # 确保输入与模型在同一设备（GPU）

        # 6. 生成回复（添加推理参数优化效果）
        outputs = model.generate(
            **inputs,
            max_new_tokens=65535,  # 最大生成 token 数
            temperature=0.7,  # 随机性（0-1，越低越确定）
            do_sample=True,  # 启用采样生成（更自然）
            pad_token_id=processor.eos_token_id  # 填充 token 指定
        )

        # 7. 解码并打印回复（只取新增生成的部分）
        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],  # 跳过输入部分
            skip_special_tokens=True  # 忽略特殊 token（如 <bos> <eos>）
        )
        return response
##########################

def AnswerImage(images:list,text:str):
    #####################
    try:
        response = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="doubao-seed-1-6-vision-250815",
        messages=[
            {
                "role": "user",
                "content": [
                        {"type": "text", "text": f"{text}"},
                ]+
                [ {"type": "image_url", "image_url": {"url": encode_image(image)}} for image in images]
                ,
            }
        ],
        )
        return (response.choices[0].message.content)
    except Exception as e:
        Debug(e)
        return AnswerImage(images,text)
    #####################
    processor=VLM.processor
    model=VLM.model
        #
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"} for _ in range(len(images))
                ]+[{"type": "text", "text": text}]
        },
    ]
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    # Generate outputs
    generated_ids = model.generate(**inputs, max_new_tokens=65535)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    #对回复进行处理
    res=generated_texts[0]
    #   
    _,res=ExtractAnswer(res)
    return res
###############################给定指令进行编辑
def EditImage(image,description:str):
    imagesResponse = client.images.generate(
        model="doubao-seededit-3-0-i2i-250628",
        prompt=description,
        image=encode_image(image),
        seed=random.randint(119,65536),
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
###############################切割区域
def ClipScore(image, target):
    processor=CLIP.processor
    model=CLIP.model
    #
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        embedding_0 = model.get_image_features(** inputs)
        inputs =processor(text=target,return_tensors="pt").to(DEVICE)
        embedding_1=model.get_text_features(**inputs)
    imageD=embedding_0.cpu().numpy().flatten()
    textD=embedding_1.cpu().numpy().flatten()
    imageD = imageD / np.linalg.norm(imageD)
    textD = textD / np.linalg.norm(textD)
    return 1.0-cosine(imageD, textD)
#粗分区域
def GroundingDINOForImage(image,target_obj:str):
    #
    processor=GroundingDINO.processor
    model=GroundingDINO.model 
    #
    inputs = processor(images=image, text=target_obj, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    #定义阈值衰减系数
    text_threshold_minus=0.05
    box_threshold_minus=0.05
    #找到结果
    def GetTarget(text_threshold,box_threshold):
        #如果任意一个阈值小于0.0直接返回None
        if text_threshold<0.0 or box_threshold<0.0:
            return None
        #
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        #裁剪出指定区域
        try:
            maxscore=0.0
            target=None
            targetbox=None
            boxes= results[0]['boxes'].cpu().tolist()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # 转为整数
                cropped_image = image.crop((x1, y1, x2, y2))
                score=ClipScore(cropped_image,target_obj)
                if score>maxscore:
                    maxscore=score
                    if maxscore>=GroundingDINOClipScoreThreold:
                        target=cropped_image
                        targetbox=[x1,y1,x2,y2]
            #如果没有结果,那么直接报错
            if target is None:
                raise Exception("未框出任何物体!")
            #
            Debug(f"GroundingDINO最大评分为:{maxscore}")
            return target,targetbox
        except Exception as e:
            return GetTarget(text_threshold-text_threshold_minus,box_threshold-box_threshold_minus)
    #
    return GetTarget(0.8,0.8)
#细分区域
def SAMForImage(image):
    generator=SAM.generator
    #
    image_array = np.array(image)
    H, W = image_array.shape[:2]
    input_boxes = [[[W//2, H//2, W//2+10, H//2+10]]]  
    outputs = generator(image, input_boxes=input_boxes)
    masks = outputs["masks"]  

    for i, mask in enumerate(masks):
        # 🔁 关键修复：Tensor → NumPy
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        mask_bool = mask_np>0.0  # 转为布尔型

        # 创建 RGBA 图像
        h, w = mask_bool.shape
        extracted = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 前景：原图 * mask
        extracted[:, :, :3] = image_array * mask_bool[:, :, np.newaxis]  # ✅ 现在都是 numpy
        # Alpha 通道
        extracted[:, :, 3] = mask_bool * 255

        # 转为 PIL 图像
        cropped_img = Image.fromarray(extracted, mode="RGBA")

        # 【可选】裁剪到物体边界
        coords = np.where(mask_bool)
        if len(coords[0]) == 0:
            continue
        top, left = coords[0].min(), coords[1].min()
        bottom, right = coords[0].max(), coords[1].max()

        # 加点 margin
        margin = 10
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(h, bottom + margin)
        right = min(w, right + margin)
        
        cropped_img = cropped_img.crop((left, top, right, bottom))
        cropped_img.save(f"tmp/object_{i}.png")
    target=int(input("请选择符合的:"))
    target_mask = masks[target]
    mask_np = target_mask.cpu().numpy() if isinstance(target_mask, torch.Tensor) else target_mask
    return mask_np>0.0
#提取区域
def ExtractByMask(image,mask):
    # 获取图像宽高
    width, height = image.size
    # 创建一个相同大小的透明图像（RGBA模式）
    result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    # 获取像素
    image_pixels = image.load()
    result_pixels = result.load()
    # 遍历每个像素
    for w in range(width):
        for h in range(height):
            # 如果遮罩为True，则保留原像素
            if mask[h, w]:  
                #保证图片肯定是rgb模式
                rgb = image_pixels[w, h]
                result_pixels[w, h] = (*rgb, 255)
    return result.convert("RGB")
            
#获取编辑后的打分
def GetImageScore(source,target,description:str):
    res=AnswerImage([source,target],description)
    score=-1
    prompt=""
    try:
        data = json.loads(res)
        if "score" in data:
            score=int(data["score"])
        if "prompt" in data:
            prompt=data["prompt"]
    except Exception as e:
        Debug(e)
        pass
    return score,prompt
#获取编辑后的局部打分
def GetImageLocalScore(source,target,description:str):
    res=GetImageScore(source,target,LoalScore_Prompt.format(description))
    #如果是因为模型框住的区域不合适，那么直接给满分即可
    if res==-1:
        Debug("所选区域有问题,直接给定满分")
        return 10
#获取编辑后的全局打分
def GetImageGlobalScore(source,target,description:str):
    return GetImageScore(source,target,GlobalScore_Prompt.format(description))
#指令优化
def OptmEditInstruction(negPrompt:str,instruction:str):
    res=AnswerText(InsOptim_Prompt.format(negPrompt,instruction))
    try:
        data=json.loads(res)
        return data["new_instruction"]
    except Exception as e:
        return ""
#艺术家打分
def GetCriticScore(source,target,instructions:list):
    instruction=""
    for idx in range(len(instructions)):
        ins=instructions[idx]
        instruction+="({})".format(idx+1)+ins+"\n"
    return GetImageScore(source,target,Critic_Prompt.format(instruction))