from transformers import AutoProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from Tips import *
from io import BytesIO
import base64
import json
##################################
VLMProcessor=None
VLMModel=None
##################################加载本地模型
def LoadVLM():
        QUANT_CONFIG = {
            "load_in_4bit": False,
            "load_in_8bit": False,
            "low_cpu_mem_usage": False,
        }
        global VLMModel,VLMProcessor
        VLMProcessor = AutoProcessor.from_pretrained("zai-org/GLM-4.1V-9B-Thinking")
        VLMModel = AutoModelForImageTextToText.from_pretrained(
            "zai-org/GLM-4.1V-9B-Thinking",
            trust_remote_code=True,  # 必须开启（多模态模型结构需远程代码）
            device_map="auto",  # 自动分配设备（优先用GPU，剩余放CPU）
            **QUANT_CONFIG  # 启用显存优化（若CPU运行，删除这一行）
            ).to(DEVICE).eval()
######################################调用本地部署模型需要提取答案
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
#####################################调用
def AnswerImage(images:list,text:str):
    #####################
    try:
        for x in images:
            if x is None:
                return "Nothing"
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
    '''
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
    '''
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
#艺术家打分
def GetCriticScore(source,target,instructions:list):
    instruction=""
    for idx in range(len(instructions)):
        ins=instructions[idx]
        instruction+="({})".format(idx+1)+ins+"\n"
    return GetImageScore(source,target,Critic_Prompt.format(instruction))