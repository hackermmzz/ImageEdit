from PIL import Image
from PIL import Image
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *
import ast
import xml.etree.ElementTree as ET
from Inpainting import *
from ImageSearchAgent import *

def Process_Inpainting(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #获取操作区域
    source,target,box=GetSoureAndTargetArea(image,task)
    Debug("GetSoureAndTargetArea:",source,",",target,",",box)
    #获取指定编辑区域
    Debug("正在获取mask区域...")
    mask=None
    if source !=None:
        mask=GroundingDINO_SAM2(image,source,box)["white_mask"]
    else:
        mask=GetBoxMask(image.size[0],image.size[1],[box])
    Debug("获取成功!")
    DebugSaveImage(mask,f"mask_{epoch}_{global_itr_cnt}.png",dir=dir)
    #
    Debug("正在进行图像编辑...")
    output_img=Inpainting(image,mask,target if target else "fill the area without any anything",neg_prompts)
    output_img=output_img.resize(image.size)
    Debug("图像编辑完成!")
    DebugSaveImage(output_img,f"edited_image_{epoch}_{global_itr_cnt}.png",dir=dir)
    return output_img

def Process_InpaintingByIpAdapter(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #获取操作区域
    source,target,box=GetSoureAndTargetArea(image,task)
    Debug("GetSoureAndTargetArea:",source,",",target,",",box)
    if  target==None:
        Debug("Process_InpaintingByIpAdapter:","target不能为None",source,target)
        return image
    #获取ip图片
    Debug("正在获取ip adapter...")
    ip_adapter=GetTargetImage(target)
    Debug("获取成功!")
    DebugSaveImage(ip_adapter,f"ip_adapter_{epoch}_{global_itr_cnt}.png",dir=dir)
    #获取指定编辑区域
    Debug("正在获取mask区域...")
    mask=None
    if source !=None:
        mask=GroundingDINO_SAM2(image,source,box)["white_mask"]
    else:
        mask=GetBoxMask(image.size[0],image.size[1],[box])
    Debug("获取成功!")
    DebugSaveImage(mask,f"mask_{epoch}_{global_itr_cnt}.png",dir=dir)
    #
    Debug("正在进行图像编辑...")
    output_img=InpaintingByIpAdapter(image,mask,target,ip_adapter,neg_prompts)
    output_img=output_img.resize(image.size)
    Debug("图像编辑完成!")
    DebugSaveImage(output_img,f"edited_image_{epoch}_{global_itr_cnt}.png",dir=dir)
    return output_img
    
def Process_ByRedMask(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #获取对象
    Debug("正在获取指定对象中...")
    target_object=GetTaskOperateObject(image,task)
    Debug("目标是:",target_object)
    #使用VLM框出指定区域
    Debug("获取编辑区域中...")
    boxes=GetROE(image,f"Please give the box of the target object.The object is:{target_object}")
    Debug("编辑区域是:",boxes)
    #获取编辑区域
    Debug("正在标记对象...")
    res=GroundingDINO_SAM2(image,target_object,boxes[0] if boxes else None)
    cutout=res["cutOut_img"]
    DebugSaveImage(cutout,f"cutout_{epoch}_{global_itr_cnt}_{RandomImageFileName()}",dir) 
    #局部补全
    return Process_Directly(cutout,f"Please work in the red area:{task}",neg_prompts,epoch,global_itr_cnt,dir)
    


def Process_Directly(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #直接编辑
    Debug("正在进行图像编辑...")
    output_img=EditImage(image,task,neg_prompts)
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
    target_object=GetTaskOperateObject(image,task)
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
    try:
        if Enabel_Agent:
            return ProcessByAgent(image,task,task_type,neg_prompts,epoch,global_itr_cnt,dir)
        #
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
    except Exception as e:
        Debug("ProcessTask:",e)
        return ProcessTask(image,task,task_type,neg_prompts,epoch,global_itr_cnt,dir)
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
###############################引入智能代理来选择如何执行
system_prompt=f'''
        You are now an image editing expert and I will give you an image and an image editing instruction, you need to complete the image editing to achieve satisfactory results with the various tools I have given you.
        You have to follow the following rules.
        (1) All images are PIL objects
        (2) You can call the following functions
            EditImageDirectly(image,prompt:str,negative_prompt_list=None) The parameters are the PIL image, the editing command, and a list of negative feedback strings, and the return value is the edited image.
            EditImageByBox(image,prompt:str,negative_prompt_list=None)The parameters are the PIL image, the editing command, and a list of negative feedback strings, and the return value is the edited image.
            EditImageByMask(image,prompt:str,negative_prompt_list=None)The parameters are the PIL image, the editing command, and a list of negative feedback strings, and the return value is the edited image.
            InpaintingByMask(image:Image.Image,prompt:str,negative_prompt_list=None) The parameters are the PIL image, the editing command, and a list of negative feedback strings, and the return value is the edited image.
            InpaintingByMaskAndIpAdapter(image:Image.Image,prompt:str,negative_prompt_list=None)The parameters are the PIL image, the editing command, and a list of negative feedback strings, and the return value is the edited image.
        (3)For the above tools, I will give specific features and application scenarios
            EditImageDirectly:This function calls the global editing model for image editing, all commands of this function are applicable, because it is purely for global editing of the image, so it must be used for global types of transformations, such as changing the screen to evening, cartooning, changing the perspective and so on, but of course it can also be used for local modifications. Its effect depends on the ability to edit the model and the accuracy of the prompt.
            EditImageByBox:This function is a call to the global editing model for image editing, it will pre-process the image, draw a rectangular box on the original image to select the area to be operated, this function is suitable for editing commands that require precise positioning (such as drawing a hat on a person's head, it will be drawn on the top of the head of the person to remind the editing model of a red area), is not suitable for global type of modification. Its effect depends on the ability of the model and the accuracy of the boxed area.
            EditImageByMask:This function is a call to the global editing model for image editing, it will pre-process the image and fill the area to be manipulated with red colour to prompt the editing model, this function is suitable for manipulating a single object and the editing commands do not depend on the original object (e.g. removing a bird from a tree, replacing a cup with a cake). Its effect depends on the capabilities of the model and the accuracy of the filled area.
            InpaintingByMask:This function is to call inpainting model for local repainting image editing, the function will automatically be from the command inside the mask area and redraw the object, this function is suitable for local modification, (such as the prompt for the cup will be removed, then the mask will be the cup, in the region of the redrawing). Its effect depends on the accuracy of the prompt and the accuracy of the mask.
            InpaintingByMaskAndIpAdapter:This function is to call the inpainting model for local repainting image editing, the function will automatically get from the command inside the appropriate mask area and inpainting objects, but in the repainting will be used to meet the prompt picture as a reference, this function is suitable for adding a single object to the picture (such as the prompt for the generation of a long-billed red crane in the sky). (for example, if the prompt is to generate a red billed crane in the sky, it will generate the crane in the appropriate area). Its effect depends on the accuracy of the prompt and the accuracy of the mask, as well as the matching nature of the ip image. But this function is better than adding objects.
        (4)Your first editing task can only call the EditImageDirectly function.
        (5)You need to make full use of the tool according to the tool I give you to get satisfactory results
        (6)Every time you edit, you need to give me the full call code of the tool you need to call <call>function call</call>.
        (7)You must give your answer according to the rule (5), no other irrelevant content is allowed.
        (8)After each call, I will give you the result, the result only contains for negative feedback (editing error, you need to adjust your editing strategy according to the negative feedback, choose the most suitable tool to complete the round of editing)
        (9) I will give you a complete workflow once. For example:
            I：The editing instruction is "change cat's color to black cross white",and the original image is "image"
            You：
                <call>EditImage(image,"change cat's color to black cross white",["low quality"])</call>
            I："background changed"
            You:  <call>EditImageByBox(image,"change cat's color to black cross white",["low quality","background changed"])</call>
            I: "..."
            ......
    '''

''''   
messages=[
    {"role":"system","content":f"{system_prompt}"},
]
'''
def ProcessByAgent(image:Image.Image,task:str,task_type:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str)->Image.Image:
    #初始化以下
    if not hasattr(THREAD_OBJECT, 'messages'):
        THREAD_OBJECT.messages=[
            {
                "role":"system",
                "content": [
                        {"type": "text", "text": f"{system_prompt}"},
                ]
            },
        ]
        THREAD_OBJECT.preImgs=None
    #   
    def Ask(image,question):
        msg=[{"type": "text", "text":question}]
        if image:
            msg+=[{"type": "image", "url": encode_image(image)}]
            #msg+=[{"type": "image_url", "image_url": {"url": encode_image(image)}}]
        THREAD_OBJECT.messages.append({"role": "user","content":msg})
    def Answer():
        return AnswerImageByPipe([],"","",THREAD_OBJECT.messages)
        '''client=client1()
        response = client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model="qwen3-vl-plus",
            messages=THREAD_OBJECT.messages,
            extra_body={
            'enable_thinking': True,
            "thinking_budget": 81920
            },
        )
        Debug("reason:",response.choices[0].message.reasoning_content)
        return (response.choices[0].message.content)  '''
    
    def EditImageDirectly(image,prompt,np=None):
        return Process_Directly(image,prompt,np,epoch,global_itr_cnt,dir)
    def EditImageByBox(image,prompt:str,np=None):
        return Process_ByBox(image,prompt,np,epoch,global_itr_cnt,dir)
    def EditImageByMask(image,prompt:str,np=None):
        return Process_ByRedMask(image,prompt,np,epoch,global_itr_cnt,dir)
    def InpaintingByMask(image:Image.Image,prompt:str,np=None):
        return Process_Inpainting(image,prompt,np,epoch,global_itr_cnt,dir)
    def InpaintingByMaskAndIpAdapter(image:Image.Image,prompt:str,np=None):
        return Process_InpaintingByIpAdapter(image,prompt,np,epoch,global_itr_cnt,dir)
    #
    if global_itr_cnt==1:
        THREAD_OBJECT.messages=THREAD_OBJECT.messages[:1]
        Ask(image.resize((512,512)), f'''the instruction is:{task},and the original image is "image"''')
    else:
        Ask(THREAD_OBJECT.preImg.resize((512,512)),"the negitive feedback of the edited image is :{neg_prompts}")
    res=Answer()
    THREAD_OBJECT.messages.append({"role": "assistant", "content": res})
    call = ET.fromstring(res).text
    Debug("This turn call is:",call)
    namespace={**globals(),**locals()}
    exec(f"edited_img={call}",namespace)
    edited_img=namespace["edited_img"].copy().convert("RGB")
    THREAD_OBJECT.preImg=edited_img
    return edited_img





################################################################
if __name__=="__main__":
    image=Image.open("image.png").convert("RGB")
    Process_Inpainting(image,"add a cat with black and white color  on the sofa ",None,1,1,"debug")