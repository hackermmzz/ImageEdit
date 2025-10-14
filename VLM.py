from transformers import AutoProcessor
from transformers import AutoProcessor, AutoModelForImageTextToText
from Tips import *
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import time
import threading
import torch
import gc
##################################
VLMProcessor=None
VLMModel=None
##################################加载本地模型
def LoadVLM():
    global VLMModel,VLMProcessor
    dir= "./Safetensors/GLM"
    VLMProcessor = AutoProcessor.from_pretrained(dir)
    VLMModel = AutoModelForImageTextToText.from_pretrained(
        dir,
        device_map=0,  # 自动分配设备（优先用GPU，剩余放CPU）
        torch_dtype=torch.bfloat16,
        ).eval()
#####################################API基础调用
def AnswerImageByAPI(images:list,role_tip:str,question:str,client,model):
    try:
        for x in images:
            if x is None:
                return None
        client_=client()
        response = client_.chat.completions.create(
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
                    [ {"type": "image_url", "image_url": {"url": encode_image(image.resize((512,512)))}} for image in images]
                    ,
                },
            ],
        )
        return (response.choices[0].message.content)
    except Exception as e:
        Debug("AnswerImageByAPI:",e)
        return AnswerImageByAPI(images,role_tip,question,client,model)
#####################################本地调用
def AnswerImageByPipe(images:list,role_tip:str,question:str,message=None):
    ##分离think和answer
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
    LoadVLM()
    #################
    global VLMModel,VLMProcessor
    messages = [
        {
            "role":"system",
            "content":[
                {"type": "text", "text": f"{role_tip}"},
            ]    
        },
        {
            "role": "user",
            "content":  [{"type": "image","url":encode_image(image.resize((512,512)))} for image in images]
                        +
                        [{"type": "text", "text": question}]
        },
    ]
    if message!=None:
        messages=message
    # Prepare inputs
    with torch.no_grad():
        inputs = VLMProcessor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = VLMModel.generate(**inputs, max_new_tokens=3000)
        outputs = VLMProcessor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    #   
    think,answer=ExtractAnswer(outputs)
    Debug("AnswerImageByPipe深度思考:",think)
    #卸载VLM
    del outputs
    del inputs
    del VLMModel 
    del VLMProcessor
    gc.collect()
    torch.cuda.empty_cache()
    #
    return answer
#####################################调用
def AnswerImage(images:list,role_tip:str,question:str,force_API=False):
    try:
        if Enable_Local_VLM and not force_API:
            return AnswerImageByPipe(images,role_tip,question)
        else:
            return AnswerImageByAPI(images,role_tip,question,client1,"qwen3-vl-plus")
    except Exception as e:
        Debug("AnswerImage:",e)
        return AnswerImage(images,role_tip,question)
##########################################获取编辑指令操作的对象
def GetTaskOperateObject(image:Image.Image,task:str):
    target_object=json.loads(AnswerImage([image],ObjectGet_Prompt,f"Now I give my edit task:{task}"))[0]
    return target_object
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
            Debug("GetImageScore:run:",e)
            return None
    tasks=[
        partial(AnswerImageByAPI,client=client1,model="qwen3-vl-plus"),#调用基础的模型
        partial(AnswerImageByAPI,client=client0,model="doubao-seed-1-6-vision-250815"),
        partial(AnswerImageByAPI,client=client0,model="doubao-seed-1-6-250615"),
    ]
    #
    cost=Timer()
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(run, tasks))  
    Debug("GetImageScore cost:",cost())
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