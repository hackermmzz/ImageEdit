from LLM import *
from VLM import *
import threading
#获取任务
def GetTask(image,description:str):
    answer=AnswerImage([image],Expert2_Prompt.format(description))
    #对answer细分
    try:
        lst=json.loads(answer)
        if type(lst)!=list:
            raise Exception("type is not list")
        return lst
    except Exception as e:
        return GetTask(image,description)
#获取编辑后的局部打分
def GetImageLocalScore(source,target,description:str):
    return 10,"","",""
    return GetImageScore([source,target],LoalScore_Prompt.format(description))
#获取编辑后的全局打分
def GetImageGlobalScore(source,target,description:str):
    return GetImageScore([source,target],GlobalScore_Prompt.format(description))
#艺术家打分
def GetCriticScore(source,target,instructions:list):
    instruction=""
    for idx in range(len(instructions)):
        ins=instructions[idx]
        instruction+="({})".format(idx+1)+ins+"\n"
    return AnswerImage([source,target],Critic_Prompt.format(instruction))
#获取场景描述
def GetDescription(image):
    return AnswerImage([image],Expert1_Prompt)   
#指令优化
def OptmEditInstruction(negPrompt:str,instruction:str):
    res=AnswerText(InsOptim_Prompt.format(negPrompt,instruction))
    try:
        data=json.loads(res)
        return data["new_instruction"]
    except Exception as e:
        return ""
##获取对象改变
def GetChange(image,task:str):
        answer=AnswerImage([image],Expert3_Prompt.format(task))
        try:
            changes=json.loads(answer)
            if type(changes)!=list:
                raise Exception("type is not list")
            return changes
        except Exception as e:
            return GetChange(image,task)