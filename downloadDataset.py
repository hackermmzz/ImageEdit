import io
import os
from datasets import load_from_disk,load_dataset,Dataset
from PIL import Image
import json
######################################下载数据集
def DownloadDataSet(save_path,url,count):
    if os.path.exists(save_path):
        return
    ds = load_dataset(url,streaming=True,split="train")
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
def Init():
    db_path="datasets/"
    url="leigangqu/VINCIE-10M"
    count=4096 
    
    db_path2="datasets2/"
    url2="pvduy/vinci_edit_sfw"
    count2=100
    #下载数据集
    DownloadDataSet(db_path,url,count)
    #加载数据
    data=LoadData(db_path)
    #拆解成数据包
    DivideData(data,"data")



#################################
if __name__=="__main__":
    Init()