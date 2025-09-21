import os
from PIL import Image
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *
from ProcessTask import *
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
    with open(dir+"tasks.txt","w",encoding="utf-8") as f:
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
            Debug("正在进行任务优化...")
            cost=Timer()
            task=polish_edit_prompt(input_img,tasks[i][0])
            Debug("指令优化耗时:",cost())
            Debug(f"优化指令为:{task}")
        #divide into four class
        output_img=None
        task_type=tasks[i][1]
        if task_type in TaskType:
            cost=Timer()
            output_img=ProcessTask(input_img,task,task_type,neg_prompts,epoch,global_itr_cnt,dir)
            Debug("指令执行耗时:",cost())
        else:
            Debug(f"unexpect task type of:{task_type} and the task is{task}")
            return
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
                
        #下一个任务
        i+=1
        global_itr_cnt=0
        input_img=max(edited_images, key=lambda x: x[0])[1]
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
    if not TEST_MODE:
        try:
            img_path=input("请输入图片路径:")
            prompt=input("请输入编辑指令:")
            ProcessImageEdit(img_path=img_path,prompt=prompt,idx=0)
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
        tasks=[]
        global TEST_CNT
        while len(target) and TEST_CNT>0:
            TEST_CNT-=1
            idx=target.pop()
            def Task(idx):
                try:
                    #创建目录
                    dir=f"{DEBUG_DIR}/{idx}/"
                    os.makedirs(dir)
                    os.makedirs(f"{dir}/Total")
                    #创建日志文件
                    if PARALLE_MODE:
                        THREAD_OBJECT.logfile=sys.stdout if (not DEBUG or not DEBUG_OUTPUT) else open(f"{dir}/debug.txt","w",encoding="utf-8")
                    #
                    target_img=f"data/{idx}/0.jpg"
                    target_prompt_file=f"data/{idx}/ins.txt"
                    if not os.path.exists(target_img) or not os.path.exists(target_prompt_file):
                        raise(f"no such {idx} file or directory!")
                    #读取指令
                    with open(target_prompt_file,"r") as f:
                        target_prompt=f.read()
                    Debug("-"*100)
                    Debug(f"第{idx}轮图像编辑开始!")
                    ProcessImageEdit(target_img,target_prompt,dir)
                    Debug(f"第{idx}轮图像处理成功!")
                except Exception as e:
                    Debug(e)
                    Debug(f"第{idx}轮图像处理失败!")
            tasks.append(partial(Task,idx=idx))
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
