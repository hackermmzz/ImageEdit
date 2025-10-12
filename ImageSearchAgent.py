from selenium import webdriver
from selenium.webdriver.common.by import By  
import requests
from PIL import Image
from Tips import *
from GroundedSam2 import *
import threading
#############################################下载图片
def DownloadImages(target:str,count:int,processImg=False)->list:
    #多线程
    thread=None
    processQue=[]
    processReady=[]
    lock=threading.Lock()
    if processImg:
        def Pop():
            lock.acquire()
            ret=None
            if len(processQue)!=0:
                ret=processQue.pop()
            lock.release()
            return ret
        def Push(ele):
            lock.acquire()
            processQue.append(ele)
            lock.release()
        def ProcessRun():
            cnt=0
            while cnt<count:
                res=Pop()
                if res!=None:
                    cnt+=1
                    res=GroundingDINO_SAM2(res,target)["original_mask"]
                    processReady.append(res)
        thread = threading.Thread(target=ProcessRun)  # args为函数参数
        thread.start()
    #
    sleepInterval=0.2
    #初始化浏览器
    edge_options = webdriver.EdgeOptions()
    edge_options.add_argument("--headless=new")
    edge_options.add_argument("--no-sandbox")
    edge_options.add_argument("--disable-infobars")
    browser = webdriver.Edge(options=edge_options)
    # 访问url
    url='''https://www.google.com.hk/search?q={}&tbm=isch'''.format(target)
    browser.get(url)
    browser.maximize_window()
    user_agent = browser.execute_script("return navigator.userAgent;")
    header={
        'User-Agent': user_agent,
        'Referer': browser.current_url, 
    }
    # 下载图片
    image_download = []
    url_download=[]
    pos = 0
    while len(image_download) < count:
        pos += 500
        # 向下滑动
        js = f'var q=document.documentElement.scrollTop={pos}'
        browser.execute_script(js)
        time.sleep(1)
        # 注意：这里使用了新的元素定位方式，替换了已废弃的find_elements_by_tag_name
        img_elements = browser.find_elements(By.TAG_NAME, 'img')
        # 遍历抓到的webElement
        for img_element in img_elements:
            img_url = img_element.get_attribute('src')
            # 过滤有效的图片URL
            if isinstance(img_url, str) and len(img_url) <= 200 and 'images' in img_url:
                if img_url not in url_download:
                    try:
                        url_download.append(img_url)
                        # 下载并保存图片
                        img=Image.open(requests.get(img_url,stream=True,headers=header).raw)
                        image_download.append(img)
                        if processImg:
                            Push(img)
                        # 防止反爬机制
                        time.sleep(sleepInterval)
                    except Exception as e:
                        Debug(f'download_images:url={img_url}')
    #关闭标签页
    browser.quit()
    #等待处理完成
    if processImg:
        thread.join()
        image_download=processReady
    #
    return image_download
###############################################选择最好的结果返回
def GetMaxFittable(images:list,target:str):
    max_score=-1.0
    target_img=None
    #
    for x in images:
        score=CLIPScore(x,target)
        if max_score<score:
            max_score=score
            target_img=x
    #
    return target_img
###############################################组合
def GetTargetImage(target:str):
    try:
        images=DownloadImages(target,10,False)
        return GetMaxFittable(images,target).convert("RGB")
    except Exception as e:
        Debug("GetTargetImage:",e)
        return GetTargetImage(target)
###############################################
if __name__ == '__main__':
    cost=Timer()
    res=DownloadImages("cat with blue eyes",10)
    print(cost())
    for x in res:
        x.save(f"cat/{RandomImageFileName()}")
        