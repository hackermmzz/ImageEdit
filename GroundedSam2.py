import argparse
import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from transformers import CLIPModel, CLIPProcessor
from scipy.spatial.distance import cosine
from Tips import *
import random
import threading
########################################################
GroundingProcessor=AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
GroundingModel=AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(DEVICE).eval()
SamModel=build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "./Safetensors/SAM/sam2.1_hiera_large.pt", device=DEVICE).eval()
SamPredictor=SAM2ImagePredictor(SamModel)
CLIPProcessor=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
CLIPLock=threading.Lock()
SAMLock=threading.Lock()
GroundingDINOLock=threading.Lock()
########################################################计算CLIP分数
def CLIPScore(image, target:str):
    try:
        CLIPLock.acquire()
        with torch.no_grad():
            inputs = CLIPProcessor(images=image, return_tensors="pt").to(DEVICE)
            embedding_0 = CLIPModel.get_image_features(** inputs)
            inputs =CLIPProcessor(text=target,return_tensors="pt").to(DEVICE)
            embedding_1=CLIPModel.get_text_features(**inputs)
        imageD=embedding_0.cpu().numpy().flatten()
        textD=embedding_1.cpu().numpy().flatten()
        imageD = imageD / np.linalg.norm(imageD)
        textD = textD / np.linalg.norm(textD)
    finally:
        CLIPLock.release()
    return 1.0-cosine(imageD, textD)
######################################################抠图
def GroundingDINO_SAM2(image,text_prompt:str):
    #运行获取sam结果和grounding结果
    def run(text_threshold:float,box_threshold:float):
        try:
            GroundingDINOLock.acquire()
            inputs = GroundingProcessor(images=image, text=text_prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = GroundingModel(**inputs)
            results = GroundingProcessor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]]
            )
        finally:
            GroundingDINOLock.release()
        # get the box prompt for SAM 2
        try:
            SAMLock.acquire()
            input_boxes = results[0]["boxes"].cpu().numpy()
            SamPredictor.set_image(np.array(image))
            masks, scores, logits = SamPredictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        finally:
            SAMLock.release()
        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))
        """
        Visualize image with supervision useful API
        """
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 创建检测结果对象
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )
        
        # 存储提取的区域
        extracted_masks_img = []    # 分割掩码图像
        extracted_boxes_img = []    # 裁剪的box区域图像
        extracted_boxes = []        # 边框坐标
        original_overlay_images = []  # 每个目标在原图上的叠加图像
        white_mask_img=[]               #原图的灰白图  
        cut_out_img=[]              #扣除对应目标的原图
        for i in range(len(detections)):
            mask = detections.mask[i]
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)

            # 原图 RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 1. 提取掩码区域图像（透明背景）
            extracted_mask = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            extracted_mask[mask] = np.concatenate([
                rgb_img[mask],
                255 * np.ones((mask.sum(), 1), dtype=np.uint8)
            ], axis=1)
            pil_mask = Image.fromarray(extracted_mask)

            # 2. 裁剪 box 图像
            box_rgb = rgb_img[y1:y2, x1:x2]
            pil_box = Image.fromarray(box_rgb)

            # 3. 构建原图大小的 overlay，只保留目标区域，其他黑色或透明
            overlay = np.zeros_like(rgb_img, dtype=np.uint8)
            overlay[mask] = rgb_img[mask]
            pil_overlay = Image.fromarray(overlay)
            # 4 
            img_arr = np.where(mask.T, 255, 0).astype(np.uint8)[..., np.newaxis]
            img_arr = np.rot90(img_arr, k=-1)
            img_arr = np.fliplr(img_arr)
            white_mask = Image.fromarray(img_arr.squeeze())  
            #5
            masked_out_rgb = rgb_img.copy()
            height, width = mask.shape
            # 遍历每个像素位置
            for i in range(height):
                for j in range(width):
                    # 检查当前位置的mask值是否为True
                    if mask[i, j]:
                        # 如果为True，将该位置的像素值设为0（或255）
                        masked_out_rgb[i, j] =[255,0,0]#[random.randint(0,255),random.randint(0,255),random.randint(0,255)]  # 或 255
            cut_out_img_ = Image.fromarray(masked_out_rgb)
            # 保存结果
            extracted_masks_img.append(pil_mask.convert("RGB"))  # 或保持透明
            extracted_boxes_img.append(pil_box.convert("RGB"))
            extracted_boxes.append((x1, y1, x2, y2))
            original_overlay_images.append(pil_overlay.convert("RGB"))
            white_mask_img.append(white_mask.convert("L"))
            cut_out_img.append(cut_out_img_.convert("RGB"))
        #从里面选取CLIP分数最高的
        maxscore=-1.0
        target_mask_image=None
        target_box_image=None
        target_box=None
        target_original_mask=None
        target_white_mask=None
        target_cut_out_img=None
        for i in range(len(extracted_boxes)):
            try:
                mask,box=extracted_masks_img[i],extracted_boxes_img[i]
                score=max(CLIPScore(mask,text_prompt),CLIPScore(box,text_prompt))
                if score>maxscore:
                    maxscore=score
                    target_box_image=box
                    target_mask_image=mask
                    target_box=extracted_boxes[i]
                    target_original_mask=original_overlay_images[i]
                    target_white_mask=white_mask_img[i]
                    target_cut_out_img=cut_out_img[i]
            except Exception as e:
                Debug("Exception:",e)
        if maxscore<0.0:
            raise Exception("None capture")
        return {
            "target_box":target_box,
            "maxscore":maxscore,
            "mask_image":target_mask_image,
            "box_image":target_box_image,
            "original_mask":target_original_mask,
            "white_mask":target_white_mask,
            "cutOut_img":target_cut_out_img,
            }
    #
    def EnsureGet(text_threshold,box_threshold):
        if text_threshold<0.0 or box_threshold<0.0:
            return None,None,None,None
        try:
            return run(text_threshold,box_threshold)
        except Exception as e:
            return EnsureGet(text_threshold-0.05,box_threshold-0.05)
    return EnsureGet(0.8,0.8)
#######################################把mask图变成纯白
def alpha_to_white_black_mask(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_img = Image.new('RGB', (width, height), color='black')
    pixels = image.load()
    new_pixels = new_img.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r != 0 or g!=0 or b!=0:
                new_pixels[x, y] = (255, 255, 255)  # 纯白
            else:
                new_pixels[x, y] = (0, 0, 0)        # 纯黑
    return new_img
#######################################
if __name__=="__main__":
    while True:
        path=input("path:")
        prompt=input("prompt:")
        res=GroundingDINO_SAM2(Image.open(path).convert("RGB"),prompt)["cutOut_img"]
        res.save("output.png")