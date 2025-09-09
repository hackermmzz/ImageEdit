import io
import os
from datasets import load_from_disk,load_dataset,Dataset
from PIL import Image
import json
from transformers.image_utils import load_image
from TopLayer import TopAgent
from Model import *
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
    #
#运行逻辑
def Run():
    scene_json='''
    {
    "global": {
        "scene_type": "natural, wooded environment",
        "background": "blurred green foliage indicating a lush forest or jungle setting with diffused natural light",
        "main_elements": "tree branches and a small bird as central subjects",
        "lighting": "natural daylight with soft, diffused illumination"
    },
    "local": {
        "objects": [
            {
                "type": "bird",
                "details": "small bird with green body plumage, red patch on head, black beak, perched on a tree branch or wooden structure"
            },
            {
                "type": "tree branches",
                "details": "thick, textured branches; one large branch in foreground with blurred (bokeh) effect, another branch the bird is on with visible wood texture"
            }
        ],
        "colors": {
            "bird": "green body, red head patch, black beak",
            "tree": "brown bark with textured surface",
            "background": "various shades of green from foliage"
        },
        "other_details": "no human presence; focus on wildlife in natural habitat with shallow depth of field emphasizing the bird"
        }
    }
    '''
    #加载所有模型
    LoadAllModel()
    #读取用户输入
    img_ptah="data/1/0.jpg"#input("请输入图像对的路径:")
    prompt="remove the bird from the tree branch while keeping the background, branch, and lighting the same"#input("请输入图像编辑描述:")
    #加载图片
    ori_image=Image.open(img_ptah).convert("RGB")
    ########################################第一层：专家池
    agent0=TopAgent()
    #专家2 任务细分
    tasks=agent0.GetTask(prompt)
    Debug("任务细分:",tasks)
    #专家1 分析图像中的场景
   # scene_json=agent0.GetDescription(ori_image)
    Debug("场景描述:",scene_json)
    #专家3 任务优化
    refine_tasks=agent0.RefineTasks(scene_json,tasks)
    Debug("任务优化:",refine_tasks)
    #专家4 获取图像变化
    changes=agent0.GetChange(scene_json,tasks)
    Debug("图像改变信息:",changes)
    ##########################################第二层：任务链
    input_img=ori_image
    i=0
    local_itr_cnt=0
    global_itr_cnt=0
    while i <len(refine_tasks):
        Debug(f"第{i+1}次任务开始!")
        ###########编辑图像
        task=refine_tasks[i]
        output=EditImage(input_img,task)
        Debug("图像编辑完成!")
        DebugSaveImage(output)
        ###########裁剪局部区域
        change=changes[i]
        #获取区域
        def GetArea(target,image,box=None):
            if target=="none":
                if box !=None:
                    x1,y1,x2,y2=map(int,box)
                    return image.crop((x1, y1, x2, y2)),box
            #GroundingDINO框出大概区域
            try:
                output,box=GroundingDINOForImage(image,target)
                DebugSaveImage(output)
                #SAM细分得到mask
    #            mask=SAMForImage(output)
                #提取区域
    #            output=ExtractByMask(output,mask)
    #            Debug(output)
                return output,box
            except Exception as e:
                Debug(e)
                return None,None
        origin,box=GetArea(change[0],input_img)
        Debug("获取原图改变区域成功!")
        edited,box=GetArea(change[1],output,box)
        Debug("获取编辑图改变区域成功!")
        #获取局部打分
        try:
            score,neg_prompt=GetImageLocalScore(origin,edited,task)
            Debug("局部打分:",score)
            if score<LocalScoreTherold and local_itr_cnt<LocalItrTherold:
                refine_tasks[i]=OptmEditInstruction(neg_prompt,task)
                local_itr_cnt+=1
                Debug(f"第{i}轮局部打分低于阈值,反向提示词为{neg_prompt}")
                continue
        except Exception as e:
            Debug(e)
        #获取全局打分
        try:
            score,neg_prompt=GetImageGlobalScore(origin,edited,task)
            Debug("全部打分:",score)
            if score<GlobalScoreTherold and global_itr_cnt<GlobalItrTherold:
                refine_tasks[i]=OptmEditInstruction(neg_prompt,task)
                global_itr_cnt+=1
                Debug(f"第{i}轮全局打分低于阈值,反向提示词为{neg_prompt}")
                continue
        except Exception as e:
            Debug(e)
        #下一个任务
        i+=1
        local_itr_cnt=0
        global_itr_cnt=0
        input_img=output
    ###################################第三层：打分
    score=GetCriticScore(ori_image,input_img,refine_tasks)
    Debug(f"最终评测机打分{score}")
    #保存图片
    fileName=RandomImageFileName()
    #input_img.save(fileName)
    DebugSaveImage(input_img,fileName)
    Debug(f"图像{fileName}保存成功!")
        
        
if __name__=="__main__":
    Init()#初始化
    Run()#运行
