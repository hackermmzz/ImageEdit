from PIL import Image
from PIL import Image
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *

def Process_Directly(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #直接编辑
    Debug("正在进行图像编辑...")
    output_img=EditImage(image,f"Edit in red boxes that {task}",neg_prompts)
    output_img=output_img.resize(image.size)
    Debug("图像编辑完成!")
    DebugSaveImage(output_img,f"edited_image_{epoch}_"+RandomImageFileName(),dir=dir)
    return output_img

def Process_Else(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    ###########编辑图像
    Debug("获取编辑区域中...")
    boxes=GetROE(image,f"Now I will give you the image-edit instruction:{task}.You should give me the fittable answer as a region for editing")
    Debug("编辑区域为:",boxes)
    img=DrawRedBox(image,boxes)
    DebugSaveImage(img,f"box_{epoch}_{global_itr_cnt}_{RandomImageFileName()}",dir)
    Debug("正在进行图像编辑...")
    output_img=EditImage(img,f"Edit in red boxes that {task}",neg_prompts)
    #将output和input缩放到同一个尺寸
    output_img=output_img.resize(image.size)
    Debug("图像编辑完成!")
    DebugSaveImage(output_img,f"edited_image_{epoch}_"+RandomImageFileName(),dir=dir)
    return output_img

def Process_Remove(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #获取移除对象
    Debug("正在获取移除的指定对象中...")
    target_object=json.loads(AnswerText(ObjectGet_Prompt,f"Now I give my edit task:{task}"))[0]
    Debug("移除的目标是:",target_object)
    #获取编辑区域
    Debug("正在标记删除对象...")
    res=GroundingDINO_SAM2(image,target_object)
    mask,cutout=res["white_mask"],res["cutOut_img"]
    DebugSaveImage(mask,f"mask_{epoch}_{global_itr_cnt}_{RandomImageFileName()}",dir)
    DebugSaveImage(cutout,f"cutout_{epoch}_{global_itr_cnt}_{RandomImageFileName()}",dir) 
    #局部补全
    Debug("正在进行inpainting...")
    output_img=EditImage(cutout,f"please work in the area marked in red:{task}",neg_prompts)
    DebugSaveImage(output_img,f"edited_image_{epoch}_"+RandomImageFileName(),dir=dir)
    return output_img

def Process_Add(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_Else(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_GlobalStyleTransfer(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_Directly(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_PerspectiveShift(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_Directly(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def ProcessTask(image:Image.Image,task:str,task_type:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str)->Image.Image:
    if task_type=="remove":
        return Process_Remove(image,task,neg_prompts,epoch,global_itr_cnt,dir)
    elif task_type=="add":
        return Process_Add(image,task,neg_prompts,epoch,global_itr_cnt,dir)
    elif task_type=="global_style_transfer":
        return Process_GlobalStyleTransfer(image,task,neg_prompts,epoch,global_itr_cnt,dir)
    elif task_type=="perspective_shift":
        return Process_PerspectiveShift(image,task,neg_prompts,epoch,global_itr_cnt,dir)
    else:
        return Process_Else(image,task,neg_prompts,epoch,global_itr_cnt,dir)
    