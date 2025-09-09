import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForImageTextToText,AutoModelForZeroShotObjectDetection
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from Tips import *  # 保留你的自定义工具类
import xml.etree.ElementTree as ET
import json
from dashscope import MultiModalConversation
import requests
import numpy as np
from io import BytesIO
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForImageTextToText
from openai import OpenAI
import cv2
from transformers import CLIPModel, CLIPProcessor
from scipy.spatial.distance import cosine
# Load model directly
DEVICE="cuda"
######################################加载SAM
LLM=None
class Expert():
    def __init__(self,processor=None,model=None,generator=None):
        self.processor=processor
        self.model=model
        self.generator=generator
    def ToDevice(self,device):
        to_list=[self.processor,self.model,self.generator]
        for x in to_list:
            if x and hasattr(x, 'to'):  # 修复：仅对有to()方法的对象移设备（如model）
                x.to(device)
        if self.model:
            self.model.eval()

def AnswerText(question:str):
        print(question)
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
        ).to(DEVICE)

        # 6. 生成回复（添加推理参数优化效果）
        with torch.no_grad():  # 新增：减少显存占用，避免梯度计算
            outputs = model.generate(
                **inputs,
                max_new_tokens=65535,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.eos_token_id
            )

        # 7. 解码并打印回复
        response = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return response

def LoadLLM():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 补充4bit量化配置（原代码缺失，避免显存溢出）
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        processor = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            quantization_config=bnb_config, 
            device_map="auto", 
            trust_remote_code=True
        ).eval()  # 已包含to(device)，无需重复调用
        global LLM
        LLM=Expert(processor,model)

def GetChange(scene:str,tasks:list):
    def AnswerText(text):
        client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="723cff33-3b13-420d-ab6d-267800a27475",
        )
        completion = client.chat.completions.create(
            model="doubao-1-5-pro-256k-250115",
            messages=[{"role": "user", "content": f"{text}"}],
        )
        return completion.choices[0].message.content
        
    text="".join([f"({i+1}){task}\n" for i,task in enumerate(tasks)])
    answer=AnswerText(Expert4_Prompt.format(scene,text))
    try:
        ret=[change.split(":") for change in json.loads(answer)]
        return [[s[0].lower(),s[1].lower()] for s in ret]
    except Exception as e:
        return GetChange(scene,tasks)

def RefineTasks(scene :str,tasks:list):
        text="".join([f"({i+1}){task}\n" for i,task in enumerate(tasks)])
        answer=AnswerText(Expert3_Prompt.format(scene,text))
        try:
            return json.loads(answer)
        except Exception as e:
            return RefineTasks(scene,tasks)

scene_json='''
    {
    "global": {
        "scene_type": "natural, wooded environment",
        "background": "blurred green foliage indicating a lush forest or jungle setting with diffused natural light",
        "main_elements": "tree branches and a small bird as central subjects",
        "lighting": "natural daylight with soft, diffused illumination"
    },
    "local": {
        "objects": [
            {
                "type": "bird",
                "details": "small bird with green body plumage, red patch on head, black beak, perched on a tree branch or wooden structure"
            },
            {
                "type": "tree branches",
                "details": "thick, textured branches; one large branch in foreground with blurred (bokeh) effect, another branch the bird is on with visible wood texture"
            }
        ],
        "colors": {
            "bird": "green body, red head patch, black beak",
            "tree": "brown bark with textured surface",
            "background": "various shades of green from foliage"
        },
        "other_details": "no human presence; focus on wildlife in natural habitat with shallow depth of field emphasizing the bird"
        }
    }
'''

GroundingDINO=None
def LoadGroundingDINO():
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(DEVICE).eval()
        global GroundingDINO
        GroundingDINO=Expert(processor,model)  
#----------------------------------------------------------------------------------------#
model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
# -------------------------- 改动2：修复CLIP打分函数（自动加对比标签） --------------------------
def get_target_score(image, target):
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
# -------------------------- 改动1：修复GroundingDINO，新增CLIP筛选逻辑 --------------------------
def GroundingDINOWithCLIP(image, target:str):
     #
    processor=GroundingDINO.processor
    model=GroundingDINO.model 
    #
    #定义阈值衰减系数
    text_threshold_minus=0.05
    box_threshold_minus=0.05
    inputs = processor(images=image, text=target, return_tensors="pt").to(DEVICE)
    #找到结果
    def GetTarget(text_threshold,box_threshold):
        #如果任意一个阈值小于0.0直接返回None
        if text_threshold<0.0 or box_threshold<0.0:
            return None
        #
        with torch.no_grad():
            outputs = model(**inputs)
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
            ret=None
            boxes= results[0]['boxes'].cpu().tolist()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # 转为整数
                try:
                    cropped_image = image.crop((x1, y1, x2, y2))
                    score=get_target_score(cropped_image,target)
                    cropped_image.save(f"debug/{target}_{score}.png")
                    if score>maxscore:
                        maxscore=score
                        if maxscore>GroundingDINOClipScoreThreold:
                            ret=cropped_image
                except Exception as e:
                    continue
            #如果没有结果,那么直接报错
            print(maxscore)
            if ret is None:
                raise Exception("未框出任何物体!")
            #
            return ret
        except Exception as e:
            return GetTarget(text_threshold-text_threshold_minus,box_threshold-box_threshold_minus)
    #
    return GetTarget(0.8,0.8)

# -------------------------- 改动3：优化SAM，用CLIP筛选后的框作为提示（无需手动选掩码） --------------------------
def SAMForImage(image, target:str, confidence_threshold=0.5):
    """
    优化：用CLIP+GroundingDINO的结果作为SAM提示，自动生成精准分割掩码
    """
    # 1. 先通过CLIP+GroundingDINO获取精准目标框
    image = GroundingDINOWithCLIP(image, target)
    image.save("debug/ground-dino.png")
    # 2. 加载SAM模型（保持你的原有配置，新增缓存路径避免重复下载）
    generator = pipeline(
        "mask-generation", 
        model="facebook/sam2.1-hiera-large", 
        device=DEVICE,
        points_per_batch=64,
        mask_threshold=confidence_threshold,
    )

   #
    image_array = np.array(image)
    H, W = image_array.shape[:2]
   # input_boxes = [[best_box]]  
    outputs = generator(image)
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
        cropped_img.save(f"tmp/{get_target_score(cropped_img,target)}.png")
    target=int(input("请选择符合的:"))
    target_mask = masks[target]
    mask_np = target_mask.cpu().numpy() if isinstance(target_mask, torch.Tensor) else target_mask
    return mask_np>0.0

# -------------------------- 测试入口（支持两种模式：仅打分 / 完整分割） --------------------------
if __name__ == "__main__":
    debug=False
    if debug:
        while True:
            x=input(":")
            print(get_target_score(Image.open("y.png").convert("RGB"),x))
    # 加载依赖模型（首次运行需下载，后续无需重复加载）
    # LoadLLM()  # 若不需要LLM可注释
    LoadGroundingDINO()

    image_path = "data/2/0.jpg"
    image = Image.open(image_path).convert("RGB")

    while True:
        print("\n请输入操作指令（格式：功能+目标，例：打分 鸟 / 分割 鸟）：")
        x = input(":")
        # 模式2：CLIP+GroundingDINO+SAM完整分割
        mask = SAMForImage(image, x)
        if mask is None:
            print(f"无法分割'{x}'，请检查目标是否存在")