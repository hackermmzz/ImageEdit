from PIL import Image
from Tips import *
from Model import *
from GroundedSam2 import *
##########################全局负反馈
def GlobalFeedback(task:str,original_image:Image,edited_image:Image,epoch:int):
    Debug("全局打分中......")
    score,neg_prompt=GetImageGlobalScore(original_image,edited_image,task)
    Debug("全局打分:",score)
    if score<GlobalScoreThershold:
        Debug(f"第{epoch}轮全局打分低于阈值,反向提示词为{neg_prompt}")
        Debug("优化指令中...")
    return score,neg_prompt
##########################修改操作
def ModifyFeedback(task:str,changes:list,original_image:Image,edited_image:Image,epoch:int,enable_local_feedback:bool,enable_global_feedback:bool,dir:str):
    #
    localScore=10
    globalScore=10
    neg_prompt=""
    #获取区域
    origin_mask,origin_box,_=GroundingDINO_SAM2(original_image,changes[0])
    Debug("获取原图改变区域成功!")
    DebugSaveImage(origin_mask,f"origin_mask_{epoch}_"+RandomImageFileName(),dir)
    DebugSaveImage(origin_box,f"origin_box_{epoch}_"+RandomImageFileName(),dir)
    edit_mask,edit_box,_=GroundingDINO_SAM2(edited_image,changes[1])
    Debug("获取编辑图改变区域成功!")
    DebugSaveImage(edit_mask,f"edit_mask_{epoch}_"+RandomImageFileName(),dir)
    DebugSaveImage(edit_box,f"edit_box_{epoch}_"+RandomImageFileName(),dir)
    #获取局部打分
    if enable_local_feedback:
        try:
            Debug("局部打分中......")
            score0,neg_prompt0=GetImageLocalScore(origin_mask,edit_mask,task)
            score1,neg_prompt1=GetImageLocalScore(origin_box,edit_box,task)
            score=max(score0,score1)
            neg_prompt_=neg_prompt0 if score0>score1 else neg_prompt1
            localScore=score
            neg_prompt=neg_prompt_
            Debug("局部打分:",localScore)
            if score<LocalScoreThershold:
                Debug(f"第{epoch}轮局部打分低于阈值,反向提示词为{neg_prompt}")
                Debug("优化指令中...")
        except Exception as e:
            Debug(e)
    #获取全局打分
    if enable_global_feedback:
        try:
            score,neg_prompt_=GlobalFeedback(task,original_image,edited_image,epoch)
            globalScore=score
            neg_prompt=neg_prompt_
        except Exception as e:
            Debug(e)
    #
    return localScore,globalScore,neg_prompt
##########################删除操作
def RemoveFeedback(task:str,removed_object:str,original_image:Image,edited_image:Image,epoch:int,enable_local_feedback:bool,enable_global_feedback:bool,dir:str):
    localScore=10
    globalScore=10
    neg_prompt=""
     #获取原图区域
    _,origin_box,box=GroundingDINO_SAM2(original_image,removed_object)
    Debug("获取原图改变区域成功!")
    DebugSaveImage(origin_box,f"origin_box_{epoch}_"+RandomImageFileName(),dir)
    #获取编辑图区域
    edited_box=edited_image.crop(box)
    Debug("获取编辑图改变区域成功!")
    DebugSaveImage(edited_box,f"edit_box_{epoch}_"+RandomImageFileName(),dir)
    #获取局部打分
    if enable_local_feedback:
        try:
            Debug("局部打分中......")
            score,neg_prompt_=GetImageLocalScore(origin_box,edited_box,task)
            localScore=score
            neg_prompt=neg_prompt_
            Debug("局部打分:",localScore)
            if score<LocalScoreThershold:
                Debug(f"第{epoch}轮局部打分低于阈值,反向提示词为{neg_prompt}")
                Debug("优化指令中...")
        except Exception as e:
            Debug(e)
    #获取全局打分
    if enable_global_feedback:
        try:
            score,neg_prompt_=GlobalFeedback(task,original_image,edited_image,epoch)
            globalScore=score
            neg_prompt=neg_prompt_
        except Exception as e:
            Debug(e)
    #
    return localScore,globalScore,neg_prompt
##########################增加操作负反馈
def AddFeedback(task:str,add_object:str,original_image:Image,edited_image:Image,epoch:int,enable_local_feedback:bool,enable_global_feedback:bool,dir:str):
    localScore=10
    globalScore=10
    neg_prompt=""
     #获取原图区域
    _,edited_box,box=GroundingDINO_SAM2(edited_image,add_object)
    Debug("获取编辑图改变区域成功!")
    DebugSaveImage(edited_box,f"origin_box_{epoch}_"+RandomImageFileName(),dir)
    #获取编辑图区域
    origin_box=original_image.crop(box)
    Debug("获取原图改变区域成功!")
    DebugSaveImage(origin_box,f"edit_box_{epoch}_"+RandomImageFileName(),dir)
    #获取局部打分
    if enable_local_feedback:
        try:
            Debug("局部打分中......")
            score,neg_prompt_=GetImageLocalScore(origin_box,edited_box,task)
            localScore=score
            neg_prompt=neg_prompt_
            Debug("局部打分:",localScore)
            if score<LocalScoreThershold:
                Debug(f"第{epoch}轮局部打分低于阈值,反向提示词为{neg_prompt}")
                Debug("优化指令中...")
        except Exception as e:
            Debug(e)
    #获取全局打分
    if enable_global_feedback:
        try:
            score,neg_prompt_=GlobalFeedback(task,original_image,edited_image,epoch)
            globalScore=score
            neg_prompt=neg_prompt_
        except Exception as e:
            Debug(e)
    #
    return localScore,globalScore,neg_prompt
##########################负反馈调用函数
def NegativeFeedback(task:str,changes:list,original_image:Image,edited_image:Image,epoch:int,enable_local_feedback:bool,enable_global_feedback:bool,dir:str):
    c0,c1=changes[0],changes[1]
    if c0!="none" and c1!="none":
        Debug("正在进行修改操作的负反馈...")
        return ModifyFeedback(task,changes,original_image,edited_image,epoch,enable_local_feedback,enable_global_feedback,dir)
    elif c0!="none" and c1=="none":
        Debug("正在进行删除操作的负反馈...")
        return RemoveFeedback(task,c0,original_image,edited_image,epoch,enable_local_feedback,enable_global_feedback,dir)
    elif c0=="none" and c1!="none":
        Debug("正在进行增加操作的负反馈...")
        return AddFeedback(task,c1,original_image,edited_image,epoch,enable_local_feedback,enable_global_feedback,dir)
    else:
        raise Exception("未知操作:",changes)