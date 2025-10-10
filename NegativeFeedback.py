from PIL import Image
from Tips import *
from Model import *
from GroundedSam2 import *
##########################全局负反馈
def GlobalFeedback(task:str,original_image:Image,edited_image:Image,epoch:int,dir:str):
    Debug("全局打分中......")
    res=GetImageGlobalScore(original_image,edited_image,task)
    score=res[0]
    Debug("全局打分:",score)
    mask=None
    if score<GlobalScoreThershold:
        Debug(f"第{epoch}轮全局打分低于阈值,反向提示词为{res[1]},正向提示词为{res[2]}")
    return res[0],res[1],res[2]
##########################负反馈调用函数
def NegativeFeedback(task:str,original_image:Image,edited_image:Image,epoch:int,dir:str):
    return GlobalFeedback(task,original_image,edited_image,epoch,dir)