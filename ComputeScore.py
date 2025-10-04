import base64
from io import BytesIO
import torch.nn as nn
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel,AutoTokenizer
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import os
import ast
from volcenginesdkarkruntime import Ark
##########################################
DEVICE=torch.device('cuda' if torch.cuda.is_available() else "cpu")
GroundingProcessor=AutoImageProcessor.from_pretrained("Safetensors/dinov2-base")
GroundingModel=AutoModel.from_pretrained("Safetensors/dinov2-base").to(DEVICE).eval()
CLIPProcessor=CLIPProcessor.from_pretrained("Safetensors/CLIP")
CLIPModel = CLIPModel.from_pretrained("Safetensors/CLIP").to(DEVICE).eval()
CLIPTokenizer = AutoTokenizer.from_pretrained("Safetensors/CLIP")
##########################################计算DINO分数
def DINOScore(source:Image.Image,target:Image.Image):
    with torch.no_grad():
        inputs1 = GroundingProcessor(images=[source], return_tensors="pt").to(DEVICE)
        inputs2 = GroundingProcessor(images=[target],return_tensors="pt").to(DEVICE)
        outputs1 = GroundingModel(**inputs1)
        outputs2 = GroundingModel(**inputs2)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0], image_features2[0]).item()
    sim = (sim + 1) / 2 
    return sim
############################################计算CLIP-I
def CLIPISCore(source:Image.Image,target:Image.Image):
    with torch.no_grad():
        source_input = CLIPProcessor(images=source, return_tensors='pt')['pixel_values'].to(DEVICE)
        target_input = CLIPProcessor(images=target, return_tensors='pt')['pixel_values'].to(DEVICE)  
        source_features = CLIPModel.get_image_features(pixel_values=source_input)
        target_features = CLIPModel.get_image_features(pixel_values=target_input)
    # 归一化
    source_features = source_features / source_features.norm(dim=1, keepdim=True)
    target_features = target_features / target_features.norm(dim=1, keepdim=True)
    # 批量计算余弦相似度
    similarity = F.cosine_similarity(source_features, target_features, dim=1).item()
    similarity=(similarity+1)/2
    return similarity
############################################计算CLIP-T
def CLIPTScore(image:Image.Image,prompt:str):
    # 特征提取
    with torch.no_grad():
        image_inputs = CLIPProcessor(images=image, return_tensors='pt').to(DEVICE)
        text_inputs = CLIPTokenizer(prompt, return_tensors='pt', max_length=200,truncation=True).to(DEVICE)
        image_features = CLIPModel.get_image_features(**image_inputs)
        text_features = CLIPModel.get_text_features(**text_inputs)
    # 归一化
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # 相似度
    similarity = F.cosine_similarity(image_features, text_features, dim=1).item()
    similarity=(similarity+1)/2
    return similarity
############################################
def AnswerImageByAPI(images:list,role_tip:str,question:str):
    def encode_image(pil_image:Image.Image)->str:
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:{'image/jpeg'};base64,{encoded_string}"
    for x in images:
        if x is None:
            return None
    client=Ark(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key="4a4becd8-195c-4fc2-b620-65cb7b72af4e",
        timeout=1800,
        # 设置重试次数
        max_retries=2,
    )
    response = client.chat.completions.create(
    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    model="doubao-1.5-vision-pro-250328",
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
############################################计算出指标
def GetIndicators(source:Image.Image,target:Image.Image,prompt:str):
    DINO=DINOScore(source,target)
    CLIPI=CLIPISCore(source,target)
    CLIPT=CLIPTScore(target,prompt)
    return DINO,CLIPI,CLIPT
############################################
def GetDescriptionForImage(image:Image.Image,tasks:list):
    system_prompt='''
        You are now an image editing expert and you need to tell me the text description of the edited image based on the image and editing instructions I give you. You need to follow the following rules.
        (1) You must give your answer in a simple but generalisable description
        (2) You must focus on my editing task and not on the image itself, e.g. although there is a dog inside the image, but I have no instructions to make changes to the dog, you simply mention that there is a dog, or even leave it out if necessary (to simplify the last sentence)
        (3) Try to keep the word count under 200 words. (But it's not mandatory.)
        (4) You must give the answer directly, in json format, specifically as follows.
            Example_1:
                input:["change the man's shirt into blue"]
                output:{
                    "prompt": "a man wearing a blue shirt with black glass"
                }
            Example_2:
                input:["higher the toy airplane to over child's head"]
                output:{
                    "prompt": "a child holding a toy airplane over his head "
                }
            Example_3:
                input:["remove the bird from the branch","add some steaks on the grill"]
                output:{
                    "prompt": "an empty branch with a man standing below,and more steaks are lay on the grill"
                }   
        (5) Don't give anything other than the answer.
        (6) The answer you give must reflects the effect of the execution of my editing instructions
    '''
    task_prompt=f"Now the image editing commands I gave are:{tasks}"
    prompt=AnswerImageByAPI([image],system_prompt,task_prompt)
    return prompt
############################################批量处理
def Process(dir:str):
    max_turn=5
    My=[[0,0,0]for _ in range(max_turn+1)]
    Vincie=[[0,0,0]for _ in range(max_turn+1)]
    Turns=[0 for _ in range(max_turn+1)]
    #
    for folder in os.listdir(dir):
        tardir=f"{dir}/{folder}"
        cnt=len(os.listdir(tardir))//2-1
        tasks=""
        with open(f"{tardir}/tasks.txt","r",encoding="utf-8")as f:
            tasks=f.read()
        tasks=ast.literal_eval(tasks)
        tasks=[x[0] for x in tasks]
        #
        if len(tasks)>max_turn:
            continue
        #
        origin=Image.open(f"{tardir}/origin.png").convert("RGB")
        my=[Image.open(f"{tardir}/my_{x}.png").convert("RGB") for x in range(cnt)]
        vincie=[Image.open(f"{tardir}/your_{x}.png").convert("RGB") for x in range(cnt)]
        for i in range(len(tasks)):
            #获取描述
            prompt=GetDescriptionForImage(origin,tasks)
            #计算我的得分
            myRes=GetIndicators(origin,my[i],prompt)
            vincieRes=GetIndicators(origin,vincie[i],prompt)
            print(myRes,"   ",vincieRes," ",prompt)
            #计算我的总分
            My[i+1]=[My[i+1][ii]+myRes[ii] for ii in range(len(myRes))]
            Vincie[i+1]=[Vincie[i+1][ii]+vincieRes[ii] for ii in range(len(vincieRes))]
            Turns[i+1]+=1
    #计算平均值
    for i in range(1,max_turn+1):
        My[i]=[x/(Turns[i] if Turns[i]!=0 else 1) for x in My[i]]
        Vincie[i]=[x/(Turns[i] if Turns[i]!=0 else 1) for x in Vincie[i]]
    #取3位小数
    for i in range(1,max_turn+1):
        My[i]=[round(x,3) for x in My[i]]
        Vincie[i]=[round(x,3) for x in Vincie[i]]
    #打印结果
    print("my: ",end='')
    for i in range(1,max_turn+1):
        print(My[i],end=' ')
    print("vincie: ",end='')
    for i in range(1,max_turn+1):
       print(Vincie[i],end=' ')
############################################
if __name__=="__main__":
    Process("C:\\Users\\mmzz\\Desktop\\compare")