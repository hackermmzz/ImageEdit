from Tips import *
from ImageEdit import *
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
#########################使用vincie数据集测试
def PredictByMagicBrush():
    #获取所有待测试的数据
    target={}
    dir="data/MagicBrush"
    trush=[125017,442231]
    for folder in trush:#os.listdir(dir):
        path=f"{dir}/{folder}"
        cnt=len(os.listdir(path))//3
        target_img=f"{path}/source_1.png"
        run_dir=f"debug/{folder}"
        task=[]
        for i in range(1,cnt+1):
            with open(f"{path}/task_{i}.txt","r",encoding="utf-8") as f:
                t=f.read()
            task.append(t)
        if cnt not in target:
            target[cnt]=[]
        target[cnt].append({"input_img":target_img,"task":task,"dir":run_dir})
    #
    target=target[3]
    #随机选择3轮次的进行编辑
    ret=[]
    for i in range(TEST_CNT):
        if len(target)==0:
            break
        idx=random.randint(0,len(target)-1)
        ret.append(target[idx])
        target=target[:idx]+target[idx+1:]
    return ret
    
###########################使用NanoBanana测试
def PredictByNanoBanana():
    '''
    {"task_type": "ic", 
     "instruction": "Change the pose of the woman in the gold bridal attire: in the original she has one hand near her face and the other across her chest, but in the edited version one hand is tucked into a hip-level pocket while the other arm is relaxed by her side. Preserve her hairstyle, makeup, jewelry, outfit and the outdoor background exactly, altering only the arm/hand positions and corresponding shadows and fabric folds for realism.", 
     "input_images": ["Nano-150k/Image/orignal/people/Advertising_1.jpg"], 
     "output_image": "Nano-150k/Image/output/action/One_Hand_in_Pocket/Advertising_1.jpg", 
     "general_prompt": "The actions after being edited are: One Hand in Pocket"
     }
     '''
    dir="Nano-150k"
    all=[]
    for folder in os.listdir(f"{dir}/json/"):
        with open(f"{dir}/json/{folder}","r",encoding="utf-8") as f:
            data=f.read()
        data=data.split("\n")
        for line in data:
            try:
                dt=json.loads(line)
                if len(dt["input_images"])==1:
                    all.append({"task":dt["instruction"],"input":dt["input_images"][0],"output":dt["output_image"]}) 
            except Exception as e:
                pass
    #随机抽取
    ret=[]
    for i in range(TEST_CNT):
        idx=random.randint(0,len(all)-1)
        target=all[idx]
        #创建目录
        dir=f"{DEBUG_DIR}/{idx}/"
        ret.append({"task":target["task"],"input_img":target["input"],"dir":dir})
    return ret
#################################
if __name__=="__main__":
    PredictByMagicBrush()