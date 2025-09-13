import os
from PIL import Image
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *
#初始化
def Init():
   pass
#运行一个单例
def ProcessImageEdit(img_path:str,prompt:str,dir="./"):
    #创建目录
    if not os.path.exists(dir):
        os.makedirs(dir)
    #加载图片
    ori_image=Image.open(img_path).convert("RGB")
    DebugSaveImage(ori_image,f"origin_image_{RandomImageFileName()}",dir)
    ########################################第一层：专家池
    #专家1 任务细分
    Debug("原指令为:",prompt)
    Debug("正在进行任务细分...")
    tasks=GetTask(ori_image,prompt)
    Debug("任务细分:",tasks)
    ##########################################第二层：任务链
    input_img=ori_image
    i=0
    local_itr_cnt=0
    global_itr_cnt=0
    neg_prompts=[]
    loop=False
    task=None
    change=None
    mask=None
    while i <len(tasks):
        epoch=i+1
        Debug(f"第{epoch}次指令编辑开始!")
        #任务优化
        if not loop:
            Debug("正在进行任务优化...")
            task=polish_edit_prompt(input_img,tasks[i])
            Debug(f"优化指令为:{task}")
        #获取执行指令后的图像改变信息
        if not loop:
            Debug("正在获取图像信息改变...")
            change=GetChange(input_img,task)
            Debug("图像改变信息:",change)
        ###########编辑图像
        Debug("正在进行图像编辑...")
        output_img=EditImage(input_img,task,neg_prompts)
        #将output和input缩放到同一个尺寸
        output_img=output_img.resize(input_img.size)
        Debug("图像编辑完成!")
        DebugSaveImage(output_img,f"edited_image_{epoch}_"+RandomImageFileName(),dir=dir)
        ###########负反馈
        res=NegativeFeedback(task,change,input_img,output_img,epoch,local_itr_cnt<LocalItrThershold,global_itr_cnt<GlobalItrThershold,dir)
        local_score=res[0]
        global_score=res[1]
        neg_prompt=res[2]
        pos_prompt=res[3]
        if local_score<LocalScoreThershold:
            neg_prompts.append(neg_prompt)
            local_itr_cnt+=1
            loop=True
            continue
        elif global_score<GlobalScoreThershold:
            neg_prompts.append(neg_prompt)
            global_itr_cnt+=1
            loop=True
            continue
        #下一个任务
        i+=1
        local_itr_cnt=0
        global_itr_cnt=0
        input_img=output_img
        neg_prompts=[]
        loop=False
    ###################################第三层：打分
    Debug("图片评分中....")
    score=GetCriticScore(ori_image,input_img,task)
    Debug(f"最终评测机打分{score}")
    #保存图片
    fileName=RandomImageFileName()
    DebugSaveImage(input_img,fileName,dir=dir)
    Debug(f"图像{fileName}保存成功!")
#运行逻辑
def Run():
    if not TEST_MODE:
        try:
            img_path=input("请输入图片路径:")
            prompt=input("请输入编辑指令:")
            ProcessImageEdit(img_path=img_path,prompt=prompt,dir="debug/")
        except Exception as e:
            print(e)
            return
    else:
        #获取所有待测试的数据
        all=[x for x in range(1,4091)]
        target=[]
        while len(all)!=0:
            x=random.randint(0,len(all)-1)
            target.append(all[x])
            all=all[:x]+all[x+1:]
        while len(target):
            idx=target.pop()
            try:
                target_img=f"data/{idx}/0.jpg"
                target_prompt_file=f"data/{idx}/ins.txt"
                if not os.path.exists(target_img) or not os.path.exists(target_prompt_file):
                    break
                #读取指令
                with open(target_prompt_file,"r") as f:
                    target_prompt=f.read()
                Debug("-"*100)
                Debug(f"第{idx}轮图像编辑开始!")
                ProcessImageEdit(target_img,target_prompt,dir=f"debug/{idx}")
                print(f"第{idx}轮图像处理成功!")
            except Exception as e:
                print(e)
                print(f"第{idx}轮图像处理失败!")
            finally:
                idx+=1
        
        
if __name__=="__main__":
    Init()#初始化
    Run()#运行
