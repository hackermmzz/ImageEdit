from PIL import Image
from PIL import Image
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *
import ast
def Process_Directly(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #直接编辑
    Debug("正在进行图像编辑...")
    output_img=EditImage(image,task,neg_prompts)
    output_img=output_img.resize(image.size)
    Debug("图像编辑完成!")
    DebugSaveImage(output_img,f"edited_image_{epoch}_{global_itr_cnt}.png",dir=dir)
    return output_img

def Process_ByBox(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    if global_itr_cnt%2==0:
        ###########编辑图像
        Debug("获取编辑区域中...")
        boxes=GetROE(image,f"Now I will give you the image-edit instruction:{task}.You should give me the fittable answer as a region for editing")
        Debug("编辑区域为:",boxes)
        image=DrawRedBox(image,boxes)
        DebugSaveImage(image,f"box_{epoch}_{global_itr_cnt}.png",dir)
    return Process_Directly(image,f"Edit in red boxes that {task}",neg_prompts,epoch,global_itr_cnt,dir)

def Process_Add(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_ByBox(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_Remove(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #获取移除对象
    Debug("正在获取移除的指定对象中...")
    target_object=json.loads(AnswerText(ObjectGet_Prompt,f"Now I give my edit task:{task}"))[0]
    Debug("移除的目标是:",target_object)
    #使用VLM框出指定区域
    Debug("获取编辑区域中...")
    boxes=GetROE(image,f"Please give the box of the target object.The object is:{target_object}")
    Debug("编辑区域是:",boxes)
    #获取编辑区域
    Debug("正在标记删除对象...")
    res=GroundingDINO_SAM2(image,target_object,boxes[0] if boxes else None)
    mask,cutout=res["white_mask"],res["cutOut_img"]
    DebugSaveImage(mask,f"mask_{epoch}_{global_itr_cnt}_{RandomImageFileName()}",dir)
    DebugSaveImage(cutout,f"cutout_{epoch}_{global_itr_cnt}_{RandomImageFileName()}",dir) 
    #局部补全
    return Process_Directly(cutout,f"Please work in the area marked in red:{task}",neg_prompts,epoch,global_itr_cnt,dir)

def Process_Replace(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_ByBox(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_GlobalStyleTransfer(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_Directly(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_PerspectiveShift(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_Directly(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_AttributeChange(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_ByBox(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_Move(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    Debug("正在获取编辑区域...")
    boxes=GetROE(image,"You should give me 2 box the first is object and the second is the region I should move to and this is my task:{task}")
    if len(boxes)!=2:
        Debug("获取移动box失败")
    else:
        Debug("获取移动box成功:",boxes)
    img=DrawRedBox(image,boxes)
    DebugSaveImage(img,f"box_{epoch}_{global_itr_cnt}.png",dir)
    return Process_Directly(img,f"Edit in red boxes that {task}",neg_prompts,epoch,global_itr_cnt,dir)

def Process_Modify(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_ByBox(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def Process_BackgroundChange(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    return Process_Directly(image,task,neg_prompts,epoch,global_itr_cnt,dir)

def ProcessTask(image:Image.Image,task:str,task_type:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str)->Image.Image:
    mp={
        "add":Process_Add,
        "remove":Process_Remove,
        "replace":Process_Replace,
        "global_style_transfer":Process_GlobalStyleTransfer,
        "perspective_shift":Process_PerspectiveShift,
        "attribute_change":Process_AttributeChange,
        "move":Process_Move,
        "modify":Process_Modify,
        "background_change":Process_BackgroundChange
    }
    
    if task_type in mp:
        fun=mp[task_type]
        return fun(image,task,neg_prompts,epoch,global_itr_cnt,dir)
    else:
        Debug(f"任务异常!任务为: {task} ,类型为: {task_type} ")
        return Process_Directly(image,task,task_type,neg_prompts,epoch,global_itr_cnt,dir)
#####################纹理修复
def TextureFix(input_img:Image.Image,edited_img:Image.Image,task:str,neg_prompts:list):
    output_img=ImageFixByAPI([input_img,edited_img],f'''fixing the right image's texture by left image and don't change or add or remove anything ''')
    return output_img,10
    #先判断哪些负反馈可以通过纹理修复进行消除
    Debug("正在获取需要修复的地方...")
    target_prompt=AnswerImage([input_img,edited_img],TextureFix_Prompt,f"My image-edit instruction is{task}.And my negtive prompts is{neg_prompts}")
    fixing=ast.literal_eval(target_prompt)
    Debug("需要修复纹理的地方:",fixing)
    #如果不需要修复
    if len(fixing)==0:
        return input_img,10
    #编辑图像
    Debug("正在修复纹理...")
    output_img=ImageFixByAPI([input_img,edited_img],f'''fixing the right image's texture by left image in the following perspective:" {fixing} " and don't change or add or delete anything ''')
    Debug("修复完成")
    Debug("正在打分...")
    score=GetImageGlobalScore(input_img,output_img,task)[0]
    Debug("打分完成")
    return output_img,score