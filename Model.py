from LLM import *
from VLM import *
import threading
from Inpainting import *
#获取任务
def GetTask(image,description:str):
    answer=AnswerImage([image],Expert1_Prompt,f"My task is:{description}")
    #对answer细分
    try:
        lst=json.loads(answer)
        if type(lst)!=list:
            raise Exception("type is not list")
        return lst
    except Exception as e:
        return GetTask(image,Expert1_Prompt,description)
#获取编辑后的全局打分
def GetImageGlobalScore(source,target,description:str):
    res=GetImageScore([source,target],GlobalScore_Prompt,"The image is as above, and my editing instruction for this round is {}".format(description))
    cost=Timer()
    pos_prompt=""
    neg_prompt=""
    if len(res[1])!=0:
        pos_prompt=SummaryPrompt(res[1])
    if len(res[2])!=0:
        neg_prompt=SummaryPrompt(res[2])
    Debug("SummaryPrompt cost:",cost())
    return res[0],pos_prompt,neg_prompt
#艺术家打分
def GetCriticScore(source,target,instruction:str):
    question=f'''
        Now I am giving my images, before editing and after editing, as shown above.
        All the editing commands are as follows {instruction}
    '''
    return AnswerImage([source,target],Critic_Prompt,question) 
#指令优化
def OptmEditInstruction(prompt:str,instruction:str):
    tip=f'''
        Now I will give my query:.
        positive prompt word:{prompt}
        Instructions for this round of editing:{instruction}
    '''
    res=AnswerText(InsOptim_Prompt,tip)
    try:
        data=json.loads(res)
        return data["new_instruction"]
    except Exception as e:
        Debug("OptmEditInstruction_Err:",res,e)
        return ""
#对反馈进行总结
def SummaryPrompt(prompts:list)->str:
    question=f'''
        Now I give my prompts:{str(prompts)}
    '''
    res=AnswerText(PromptFeedbackSummary_Prompt,question)
    try:
        data=json.loads(res)
        return data["prompt"]
    except Exception as e:
        Debug("SummaryPrompt:",e,res)
        return SummaryPrompt(prompts)
##########################################
if __name__=="__main__":
    img=Image.open("data/1/0.jpg").convert("RGB")
    GetImageGlobalScore(img,img,"remove the bird")