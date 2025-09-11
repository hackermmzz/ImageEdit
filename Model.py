from LLM import *
from VLM import *
#获取任务
def GetTask(description:str):
        answer=AnswerText(Expert2_Prompt.format(description))
        #对answer细分
        try:
            lst=json.loads(answer)
            if type(lst)!=list:
                raise Exception("type is not list")
            return lst
        except Exception as e:
            return GetTask(description)
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
def  RefineTasks(scene :str,tasks:list):
        #
        text=""
        for i in range(len(tasks)):
            text+="({})".format(i+1)+tasks[i]+"\n"
        answer=AnswerText(Expert3_Prompt.format(scene,text))
        #
        try:
            lst=json.loads(answer)
            if type(lst)!=list:
                raise Exception("type is not list")
            refined_tasks=lst
            return refined_tasks
        except Exception as e:
            return RefineTasks(scene,tasks)#无限循环直到正确
##获取对象改变
def GetChange(scene:str,tasks:list):
        text=""
        for i in range(len(tasks)):
            text+="({})".format(i+1)+tasks[i]+"\n"
        answer=AnswerText(Expert4_Prompt.format(scene,text))
        try:
            ret=[]
            changes=json.loads(answer)
            if type(changes)!=list:
                raise Exception("type is not list")
            for change in changes:
                s=change.split(":")
                ret.append([s[0].lower(),s[1].lower()])
            return ret
        except Exception as e:
            return GetChange(scene,tasks)