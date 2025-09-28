from Tips import *
from ImageEdit import *
import random
from concurrent.futures import ThreadPoolExecutor
from functools import partial
#########################使用vincie数据集测试
def PredictByVINCIE():
    #获取所有待测试的数据
    target=[]
    for i in range(TEST_CNT):
        idx=random.randint(1,4091)
        try:
            #创建目录
            dir=f"{DEBUG_DIR}/{idx}/"
            #
            target_img=f"data/{idx}/0.jpg"
            target_prompt_file=f"data/{idx}/ins.txt"
            if not os.path.exists(target_img) or not os.path.exists(target_prompt_file):
                raise(f"no such {idx} file or directory!")
            #读取指令
            with open(target_prompt_file,"r") as f:
                target_prompt=f.read()
        except Exception as e:
            pass
        target.append({"input_img":target_img,"task":target_prompt,"dir":dir})
    return target
    
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
        try:
           ret.append({"task":target["instruction"],"input_img":target["input_images"][0],"dir":dir})
        except Exception as e:
            pass
    return ret
#################################
if __name__=="__main__":
    PredictByNanoBanana()