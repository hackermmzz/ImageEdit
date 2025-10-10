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
def Process_InpaintingByIpAdapter(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #获取操作区域
    source,target,box=GetSoureAndTargetArea(image,task)
    if  target==None:
        Debug("Process_InpaintingByIpAdapter:","target不能为None",source,target)
        return image
    #获取ip图片
    ip_adapter=GetTargetImage(target)
    DebugSaveImage(ip_adapter,f"ip_adapter_{epoch}_{global_itr_cnt}.png",dir=dir)
    #获取指定编辑区域
    Debug("正在获取mask区域...")
    mask=None
    if source !=None:
        mask=GroundingDINO_SAM2(image,source,[box])["cutOut_img"]
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
    
def Process_Inpainting(image:Image.Image,task:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str):
    #获取操作对象
    Debug("正在获取指定对象中...")
    target_object=GetTaskOperateObject(image,task)
    Debug("操作的目标是:",target_object)
    res=GroundingDINO_SAM2(image,target_object, None)
    mask=res["white_mask"]
    DebugSaveImage(mask,f"mask_{epoch}_{global_itr_cnt}_{RandomImageFileName()}",dir)
    #
    Debug("正在进行图像编辑...")
    output_img=Inpainting(image,mask,task,neg_prompts)
    output_img=output_img.resize(image.size)
    Debug("图像编辑完成!")
    DebugSaveImage(output_img,f"edited_image_{epoch}_{global_itr_cnt}.png",dir=dir)
    return output_img

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
        你现在是一个图像编辑专家，我将给你一个图像和一个图像编辑指令，你需要凭借我给你的各种工具完成图像编辑达到满意的效果。
        你必须遵循以下规则。
        (1)所有的图像均是PIL对象
        (2)你可以调用以下函数
            EditImageDirectly(image,prompt:str,negative_prompt_list=None) 参数为PIL图像，编辑指令，以及负反馈字符串列表,返回值是编辑好的图像
            EditImageByBox(image,prompt:str,negative_prompt_list=None)参数为PIL图像，编辑指令，以及负反馈字符串列表,返回值是编辑好的图像
            EditImageByMask(image,prompt:str,negative_prompt_list=None)参数为PIL图像，编辑指令，以及负反馈字符串列表,返回值是编辑好的图像
            InpaintingByMask(image:Image.Image,prompt:str,negative_prompt_list=None) 参数为PIL图像，编辑指令,以及负反馈字符串列表,返回值是编辑好的图像
            InpaintingByMaskAndIpAdapter(image:Image.Image,prompt:str,negative_prompt_list=None)参数为PIL图像，编辑指令,以及负反馈字符串列表,返回值是编辑好的图像
        (3)对于以上工具，我将给出具体功能和适用场景
            EditImageDirectly:这个函数是调用的全局编辑模型进行图像编辑，这个函数所有指令都适用，因为是单纯对图像进行全局编辑，所以全局类型的变换必须使用它，比如把画面改成傍晚，卡通画，视角改变等等，当然局部修改也可以使用它.它的效果依赖于编辑模型的能力和prompt的精确度
            EditImageByBox:这个函数是调用的全局编辑模型进行图像编辑,它会预处理一下图片，在原图上画出一个矩形框，来框选要操作的区域，这个函数适用于需要精准定位的编辑指令（如在人头部画一个帽子，便会在人头顶画一个红色区域以此提醒编辑模型），不适合全局类型的修改。它的效果依赖于模型的能力以及框选区域的准确性。
            EditImageByMask:这个函数是调用的全局编辑模型进行图像编辑,它会预处理一下图片，使用红色填充满要操作的区域提示编辑模型，这个函数适合用于操作单个物体且编辑指令不依赖原物体（如移除树上的鸟、将杯子替换为蛋糕）。它的效果依赖于模型的能力以及填充区域的准确性。
            InpaintingByMask:这个函数是调用inpainting模型进行局部重绘的图像编辑，函数会自动将从指令里面获取mask区域和重新绘制的object，这个函数适合用于局部修改，（如prompt为将杯子移除，那么mask就会为杯子，在该区域进行重新绘制）。它的效果依赖于prompt的准确性以及mask的准确性。
            InpaintingByMaskAndIpAdapter:这个函数是调用inpainting模型进行局部重绘的图像编辑，函数会自动将从指令里面获取合适的mask区域和inpainting的物体，但是在重绘的时候会使用一张符合prompt图片作为参照,这个函数适合用于向图片内添加单个物体（比如prompt为在天空生成一只红色长嘴丹顶鹤，它便会在合适区域去生成丹顶鹤）。它的效果依赖于prompt的准确性以及mask的准确性，以及ip图片的匹配性质。但是相对于增添物体，这个函数更好。
        (4)你第一次的编辑任务只能调用EditImageDirectly函数。
        (5)你需要根据我给你的工具，充分利用好工具，得到满意的效果
        (6)每次编辑，你需要给我你需要调用工具的完整调用代码<call>function call</call>
        (7)必须按照第(5)个规则给出你的回答,不准出现其他任何无关内容
        (8)每次调用完，我都会给你结果，结果只包含为负反馈（编辑出错的地方，你需要根据负反馈去调整你的编辑策略，选择最合适的工具完成本轮编辑
        (9)我将给你一次完整的工作流程。例如：
            我：编辑指令是"change cat's color to black cross white",原始图像是image
            你：
                <call>EditImage(image,"change cat's color to black cross white",["low quality"])</call>
            我："background changed"
            你:  <call>EditImageByBox(image,"cat","change cat's color to black cross white",["low quality","background changed"])</call>
            我: "..."
            ......
    '''

''''   
messages=[
    {"role":"system","content":f"{system_prompt}"},
]
'''
messages=[
        {
            "role":"system",
            "content": [
                    {"type": "text", "text": f"{system_prompt}"},
            ]
        },
    ]
preImg=None
def ProcessByAgent(image:Image.Image,task:str,task_type:str,neg_prompts:list,epoch:int,global_itr_cnt:int,dir:str)->Image.Image:
    #
    global preImg
    def Ask(image,question):
        #messages.append({"role": "user", "content": f'''指令是:{task},原始图像是 image'''})
        msg=[{"type": "text", "text":question}]
        if image:
            msg+=[{"type": "image_url", "image_url": {"url": encode_image(image)}}]
        messages.append({"role": "user","content":msg})
    def Answer():
        client=client1()
        response = client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model="qwen3-vl-plus",
            messages=messages,
        )
        return (response.choices[0].message.content)
        client = Ark(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
            api_key="da11cd64-e1ac-452f-8982-238770638e98",
        )
        completion = client.chat.completions.create(
            # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
            model="deepseek-v3-250324",
            messages=messages
        )
        return completion.choices[0].message.content    
    
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
    global messages
    if global_itr_cnt==1:
        messages=messages[:1]
        Ask(image, f'''指令是:{task},原始图像是 image''')
    else:
        Ask(preImg,"负反馈:{neg_prompts}")
    res=Answer()
    messages.append({"role": "assistant", "content": res})
    call = ET.fromstring(res).text
    Debug("本次调用为:",call)
    namespace={**globals(),**locals()}
    exec(f"edited_img={call}",namespace)
    edited_img=namespace["edited_img"].copy().convert("RGB")
    preImg=edited_img
    return edited_img





################################################################
if __name__=="__main__":
    image=Image.open("image.png").convert("RGB")
    Process_InpaintingByIpAdapter(image,"add a cat with black and white color  on the sofa ",None,1,1,"debug")