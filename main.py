import os
from PIL import Image
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *
from ProcessTask import *
from Predict import *
#初始化
def Init():
   pass
#运行一个单例
def ProcessImageEdit(img_path:str,prompt:str,dir:str):
    #
    cost_total=Timer()
    #加载图片
    ori_image=Image.open(img_path).convert("RGB")
    DebugSaveImage(ori_image,f"origin_image_{RandomImageFileName()}",dir)
    ########################################第一层：专家池
    #专家1 任务细分
    Debug("原指令为:",prompt)
    Debug("正在进行任务细分...")
    cost=Timer()
    tasks=GetTask(ori_image,prompt)
    Debug("任务细分耗时:",cost())
    Debug("任务细分:",tasks)
    ##########################################写入文件
    with open(dir+"/tasks.txt","w",encoding="utf-8") as f:
        f.write(str(tasks))
    ##########################################第二层：任务链
    input_img=ori_image
    i=0
    global_itr_cnt=0
    neg_prompts=[]
    edited_images=[]
    task=None
    EpochBestImage=[]
    while i <len(tasks):
        epoch=i+1
        global_itr_cnt+=1
        Debug(f"第{epoch}次指令编辑,第{global_itr_cnt}次尝试开始!")
        #任务优化
        if global_itr_cnt==1:
            if Enable_TaskPolish:
                Debug("正在进行任务优化...")
                cost=Timer()
                task=polish_edit_prompt(input_img,tasks[i][0])
                Debug("指令优化耗时:",cost())
                Debug(f"优化指令为:{task}")
            else:
                task=tasks[i][0]
        #任务处理
        output_img=None
        task_type=tasks[i][1]
        cost=Timer()
        #获取任务处理
        output_img=ProcessTask(input_img,task,task_type,neg_prompts,epoch,global_itr_cnt,dir)
        Debug("指令执行耗时:",cost())
        ###########负反馈
        cost=Timer()
        res=NegativeFeedback(task,input_img,output_img,global_itr_cnt,dir)
        Debug("负反馈耗时:",cost())
        global_score=res[0]
        neg_prompt=res[1]
        pos_prompt=res[2]
        edited_images.append((global_score,output_img))
        if  global_score<GlobalScoreThershold:
            if global_itr_cnt<GlobalItrThershold:
                cost=Timer()
                Debug("正在优化指令...")
                task=OptmEditInstruction(pos_prompt,task)
                Debug(f"优化完成!指令为\"{task}\"")
                Debug("指令优化耗时:",cost())
                neg_prompts.append(neg_prompt)
                continue
        #获取得分最高的图片
        target_score,target_img=max(edited_images, key=lambda x: x[0])
        #修复一下纹理损失，避免多轮下来纹理损失过度积累
        if Enable_TextureFix:
            cost=Timer()
            Debug("正在进行纹理修复...")
            res,score=TextureFix(input_img,target_img,task,neg_prompts)
            Debug("纹理修复耗时:",cost())
            DebugSaveImage(res,f"fixing_{epoch}.png",dir)
            Debug("修复后全局打分为:",score)
            if target_score<score:
                target_img=res
        #下一个任务
        input_img=target_img
        i+=1
        global_itr_cnt=0
        neg_prompts=[]
        edited_images=[]
        EpochBestImage.append(input_img)
        DebugSaveImage(input_img,f"epoch{epoch}_edited_image.png",dir=dir)
    ###################################第三层：打分
    Debug("图片评分中....")
    cost=Timer()
    score=GetCriticScore(ori_image,input_img,prompt)
    Debug("图片评分耗时:",cost())
    Debug(f"最终评测机打分{score}")
    #保存图片
    fileName=RandomImageFileName()
    DebugSaveImage(input_img,fileName,dir=dir)
    Debug(f"图像{fileName}保存成功!")
    #保存所有轮最好的图片
    for x in range(len(EpochBestImage)):
        img=EpochBestImage[x]
        DebugSaveImage(img,f"{x}.png",dir+"/Total/")
    #统计一轮整体耗时
    Debug("整体耗时:",cost_total())
#运行逻辑
def Run():
    data=None
    if not TEST_MODE:
        try:
            img_path=input("请输入图片路径:")
            prompt=input("请输入编辑指令:")
            data=[{"input_img":img_path,"task":prompt,"dir":f"{DEBUG_DIR}/1"}]
        except Exception as e:
            print(e)
            return
    else:
       # data=PredictByVINCIE()
        data=PredictByNanoBanana()
    #构建任务
    tasks=[]
    def Task(input_img:str,tasks:str,dir:str):
        os.makedirs(dir)
        os.makedirs(f"{dir}/Total")
        if PARALLE_MODE:
            THREAD_OBJECT.logfile=open(f"{dir}/debug.txt","w",encoding="utf-8")
        try:
            Debug("处理开始")
            ProcessImageEdit(input_img,tasks,dir)
            Debug("处理成功")
        except Exception as e:
            Debug("处理失败:",e)
    for x in data:
        tasks.append(partial(Task,input_img=x["input_img"],tasks=x["task"],dir=x["dir"]))
    #如果并行，则开启多线程
    if PARALLE_MODE:
        with ThreadPoolExecutor(max_workers=min(len(tasks),65535)) as executor:
            futures = [executor.submit(task) for task in tasks]
    else:
        for task in tasks:
            task()
if __name__=="__main__":
    Init()#初始化
    Run()#运行
