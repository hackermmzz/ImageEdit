import torch
######################################
from Tips import *
import json
from Model import AnswerText,AnswerImage
######################################指定设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
######################################


class TopAgent():
    def __init__(self):
        pass
    def __call__(self,image,description):
        #获取图片整体描述
        res=self.GetDescription(image)
        
        return res
    #获取对场景的描述
    def GetDescription(self,image):
        return AnswerImage([image],Expert1_Prompt)   
    #获取所有的编辑任务
    def GetTask(self,description:str):
        answer=AnswerText(Expert2_Prompt.format(description))
        #对answer细分
        try:
            lst=json.loads(answer)
            if type(lst)!=list:
                raise Exception("type is not list")
            return lst
        except Exception as e:
            return self.GetTask(description)
    #细化任务
    def  RefineTasks(self,scene :str,tasks:list):
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
            return self.RefineTasks(scene,tasks)#无限循环直到正确
    #获取指令执行后的改变
    def GetChange(self,scene:str,tasks:list):
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
            return self.GetChange(scene,tasks)