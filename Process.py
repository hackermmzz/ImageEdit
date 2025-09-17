from PIL import Image
from PIL import Image
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *


def ProcessTask(image:Image.Image,task:str,task_type:str,epoch:int,dir:str):
    ###########编辑图像
    Debug("获取编辑区域中...")
    boxes=GetROE(image,f"Now I will give you the image-edit instruction:{task}.You should give me the fittable answer as a region for editing")
    Debug("编辑区域为:",boxes)
    img=DrawRedBox(image,boxes)
    DebugSaveImage(img,f"box_{epoch}_{global_itr_cnt}_{RandomImageFileName()}")
    Debug("正在进行图像编辑...")
    output_img=EditImage(img,f"Edit in red boxes that {task}",neg_prompts,True)
    #将output和input缩放到同一个尺寸
    output_img=output_img.resize(image.size)
    Debug("图像编辑完成!")
    DebugSaveImage(output_img,f"edited_image_{epoch}_"+RandomImageFileName(),dir=dir)
    ###########负反馈
    inpainting_img=NegativeFeedback(task,image,output_img,global_itr_cnt,dir)
    global_score=inpainting_img[0]
    neg_prompt=inpainting_img[1]
    pos_prompt=inpainting_img[2]
    edited_images.append((global_score,output_img))
    if  global_score<GlobalScoreThershold:
        if global_itr_cnt<GlobalItrThershold:
            Debug("正在优化指令...")
            task=OptmEditInstruction(pos_prompt,task)
            Debug(f"优化完成!指令为\"{task}\"")
            neg_prompts.append(neg_prompt)
            loop=True
            continue
        #否则对区域重新绘制以及调用图像编辑API接口
        else:
            #如果任务是移除，那么直接调用inpainting
            if task_type=="remove":
                Debug("inpainting......")
                inpainting_img=InpaintingArea(image,task)
                DebugSaveImage(inpainting_img,f"inpainting_{epoch}_{RandomImageFileName()}",dir)
                Debug("全局打分中......")
                score=GetImageGlobalScore(image,inpainting_img,task)[0]
                Debug("全局打分:",score)
                edited_images.append((score,inpainting_img))
            #调用API接口
            Debug("获取编辑区域中...")
            boxes=GetROE(image,f"Now I will give you the image-edit instruction:{task}.You should give me the fittable answer as a region for editing")
            Debug("编辑区域为:",boxes)
            img=DrawRedBox(image,boxes)
            DebugSaveImage(img,f"boxed_edit_{epoch}.png",dir)
            Debug("API绘制中...")
            output_img=EditImage(img,task,neg_prompts,byAPI=True)
            Debug("API绘制完成")
            DebugSaveImage(output_img,f"Edit_By_Api_{epoch}.png",dir)
            Debug("全局打分中......")
            score=GetImageGlobalScore(image,inpainting_img,task)[0]
            Debug("全局打分:",score)
            edited_images.append((score,output_img))
            
    #下一个任务
    i+=1
    global_itr_cnt=0
    image=max(edited_images, key=lambda x: x[0])[1]
    neg_prompts=[]
    edited_images=[]
    loop=False
    DebugSaveImage(image,f"epoch{epoch}_edited_image.png",dir=dir)