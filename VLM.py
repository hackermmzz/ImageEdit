from transformers import AutoProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from Tips import *
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
#####################################API基础调用
def ImageAnswer(images:list,role_tip:str,question:str,client,model):
    try:
        for x in images:
            if x is None:
                return None
        client=client()
        response = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model=model,
        messages=[
            {
                "role":"system",
                "content": [
                        {"type": "text", "text": f"{role_tip}"},
                ]
            },
            {
                "role": "user",
                "content": [
                        {"type": "text", "text": f"{question}"},
                ]+
                [ {"type": "image_url", "image_url": {"url": encode_image(image)}} for image in images]
                ,
            }
        ],
        )
        return (response.choices[0].message.content)
    except Exception as e:
        Debug("Answer_Image:",e)
        return ImageAnswer(images,role_tip,question,client,model)
#####################################调用
def AnswerImage(images:list,role_tip:str,question:str):
    return ImageAnswer(images,role_tip,question,client0,"doubao-seed-1-6-vision-250815")
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
##########################################获取ROE
def GetROE(image:Image.Image,question:str) ->list:
    try:    
        res=AnswerImage([image],GetMaskArea_Prompt,question)
        rr=""
        for x in res:
            if x not in "[]()":
                rr=rr+x
        res=[float(s.strip()) for s in rr.split(",")]
        w,h=image.size
        ret=[]
        tmp=[]
        for i in range(len(res)):
            val=res[i]
            if i%4==0:
                val=int(val*w)
            elif i%4==1:
                val=int(val*h)
            elif i%4==2:
                val=int(val*w)
            else:
                val=int(val*h)
            tmp.append(val)
            if i%4==3:
                ret.append(tmp)
                tmp=[]
    except Exception as e:
        Debug("GetROE:",e,res)
        return GetROE(image,question)
    return ret
##########################################获取得分
def GetImageScore(images:list,role_tip:str,question:str):
    def run(task):
        try:
            return task(images=images,
                        role_tip=role_tip,
                        question=question
                        )
        except Exception as e:
            Debug("GetImageScore:",e)
            return None
    tasks=[
        partial(ImageAnswer,client=client0,model="doubao-seed-1-6-vision-250815"),#调用基础的模型
        partial(ImageAnswer,client=client1,model="qwen-vl-max"),
        partial(ImageAnswer,client=client1,model="qwen-vl-plus-latest"),
    ]
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        results = executor.map(run, tasks)
    #解析结果
    useful=[]
    for res in results:
        try:
            score=-1
            negative_prompt=""
            positive_prompt=""
            data = json.loads(res)
            if "score" in data:
                score=int(data["score"])
            if "negative_prompt" in data:
                negative_prompt=data["negative_prompt"]
            if "positive_prompt" in data:
                positive_prompt=data["positive_prompt"]
            useful.append((score,negative_prompt,positive_prompt))
        except Exception as e:
            pass
    #选取最小得分作为最终得分
    total_score=0
    target_negative_prompt=[]
    target_positive_prompt=[]
    for score,negative_prompt,positive_prompt in useful:
        total_score=total_score+score
        target_negative_prompt.append(negative_prompt)
        target_positive_prompt.append(positive_prompt)
    #
    return total_score/len(useful),target_negative_prompt,target_positive_prompt