import io
import os
from datasets import load_from_disk,load_dataset,Dataset
from PIL import Image
import json
from GroundedSam2 import*
from ImageEdit import*
from LLM import *
from VLM import *
from Model import *
from NegativeFeedback import *
######################################下载数据集
def DownloadDataSet(save_path,count=4096):
    if os.path.exists(save_path):
        return
    ds = load_dataset("leigangqu/VINCIE-10M",streaming=True,split="train")
    ds=list(ds.take(count))
    ds=Dataset.from_list(ds)
    ds.save_to_disk(save_path)
######################################获取数据
def LoadData(path):
    dataset = load_from_disk(dataset_path=path)#加载数据集
    data=[]
    for item in dataset:
        try:
            total_frames = len(item['image'])
            total_frames = len(item['image'])
            split_point = total_frames // 2
            #获取两个图像
            input_frames = item['image'][:split_point]
            output_frames = item['image'][split_point:]
            #获取指令描述
            description = json.loads(item["ann_v0"])[0]["text"]["summary_change"]
            #
            sj={"input_img":input_frames,"description":description,"output_img":output_frames}
            #
            data.append(sj)
        except Exception as e:
            pass
    #
    return data
###################################拆分数据
def DivideData(data,path,debug=False):
    if os.path.exists(path)==False:
        os.mkdir(path)
    elif not debug:
        return
    #
    idx=0
    for ele in data:
        idx+=1
        sub_folder=path+"/"+str(idx)
        if os.path.exists(sub_folder)==False:
            os.mkdir(sub_folder)
        #
        def Save(idx,img_data):
            img=Image.open(io.BytesIO(img_data))
            img.save(sub_folder+"/"+f"{idx}.jpg")

        Save(0,ele["input_img"][0])
        Save(1,ele["output_img"][0])
        with open(sub_folder+"/ins.txt",mode="w",encoding="utf-8") as f:
            f.write(ele["description"])
#初始化
def Init():
    db_path="datasets/"
    #下载数据集
    DownloadDataSet(db_path)
    #加载数据
    data=LoadData(db_path)
    #拆解成数据包
    DivideData(data,"data")
#运行一个单例
def ProcessImageEdit(img_path:str,prompt:str,dir="./"):
    #创建目录
    if not os.path.exists(dir):
        os.makedirs(dir)
    #加载图片
    ori_image=Image.open(img_path).convert("RGB")
    ########################################第一层：专家池
    #专家1 分析图像中的场景
    Debug("正在获取场景描述...")
    scene_json=GetDescription(ori_image)
    Debug("场景描述:",scene_json)
    #专家2 任务细分
    Debug("正在进行任务细分...")
    tasks=GetTask(prompt)
    Debug("任务细分:",tasks)
    #专家3 获取图像变化
    Debug("正在获取图像信息改变...")
    changes=GetChange(scene_json,tasks)
    Debug("图像改变信息:",changes)
    ##########################################第二层：任务链
    input_img=ori_image
    i=0
    local_itr_cnt=0
    global_itr_cnt=0
    neg_prompts=[]
    while i <len(tasks):
        epoch=i+1
        Debug(f"第{epoch}次指令编辑开始!")
        #任务优化
        Debug("正在进行任务优化:")
        task=polish_edit_prompt(input_img,tasks[i])
        Debug("优化指令为:{}".format(task))
        ###########编辑图像
        Debug("正在进行图像编辑...")
        output_img=EditImage(input_img,task,neg_prompts)
        #将output和input缩放到同一个尺寸
        output_img=output_img.resize(input_img.size)
        Debug("图像编辑完成!")
        DebugSaveImage(output_img,f"edited_image_{epoch}_"+RandomImageFileName(),dir=dir)
        ###########负反馈
        local_score,global_score,neg_prompt=NegativeFeedback(task,changes[i],input_img,output_img,epoch,local_itr_cnt<LocalItrThershold,global_itr_cnt<GlobalItrThershold,dir)
        if local_score<LocalScoreThershold:
            neg_prompts.append(neg_prompt)
            local_itr_cnt+=1
            continue
        elif global_score<GlobalScoreThershold:
            neg_prompts.append(neg_prompt)
            global_itr_cnt+=1
            continue
        #下一个任务
        i+=1
        local_itr_cnt=0
        global_itr_cnt=0
        input_img=output_img
        neg_prompts=[]
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
        idx=1
        while True:
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
