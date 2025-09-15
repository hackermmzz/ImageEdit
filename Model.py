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
    return GetImageScore([source,target],GlobalScore_Prompt,"The image is as above, and my editing instruction for this round is {}".format(description))
#艺术家打分
def GetCriticScore(source,target,instructions:list):
    instruction=""
    for idx in range(len(instructions)):
        ins=instructions[idx]
        instruction+="({})".format(idx+1)+ins+"\n"
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


##########################################
if __name__=="__main__":
    x=input(":")
    print(OptmEditInstruction("add some uncooked steaks","add some steaks"))