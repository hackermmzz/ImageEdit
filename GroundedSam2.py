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
########################################################
GroundingProcessor=AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
GroundingModel=AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEVICE).eval()
SamModel=build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "./checkpoints/sam2.1_hiera_large.pt", device=DEVICE).eval()
SamPredictor=SAM2ImagePredictor(SamModel)
CLIPProcessor=CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
########################################################计算CLIP分数
def CLIPScore(image, target:str):
    with torch.no_grad():
        inputs = CLIPProcessor(images=image, return_tensors="pt").to(DEVICE)
        embedding_0 = CLIPModel.get_image_features(** inputs)
        inputs =CLIPProcessor(text=target,return_tensors="pt").to(DEVICE)
        embedding_1=CLIPModel.get_text_features(**inputs)
    imageD=embedding_0.cpu().numpy().flatten()
    textD=embedding_1.cpu().numpy().flatten()
    imageD = imageD / np.linalg.norm(imageD)
    textD = textD / np.linalg.norm(textD)
    return 1.0-cosine(imageD, textD)
######################################################抠图
def GroundingDINO_SAM2(image,text_prompt:str):
    #运行获取sam结果和grounding结果
    def run(text_threshold:float,box_threshold:float):
        SamPredictor.set_image(np.array(image))
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
        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()
        masks, scores, logits = SamPredictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
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
        extracted_masks = []    # 存储分割出的区域
        extracted_boxes = []    # 存储带边界框的区域
        
        # 遍历每个检测结果，提取对应的区域
        for i in range(len(detections)):
            # 获取单个检测的掩码和边界框
            mask = detections.mask[i]
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            
            # 1. 提取分割区域（带透明背景）
            # 创建一个与原图相同大小的透明图像
            extracted_mask = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            
            # 将原图的BGR转换为RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将原图像素复制到提取区域，掩码区域可见，其他区域透明
            extracted_mask[mask] = np.concatenate([
                rgb_img[mask],  # RGB通道
                255 * np.ones((mask.sum(), 1), dtype=np.uint8)  # Alpha通道（不透明）
            ], axis=1)
            
            # 转换为PIL Image并添加到结果列表
            pil_mask = Image.fromarray(extracted_mask)
            #
            box_img = img.copy()
            # 转换为RGB并裁剪边界框区域
            box_rgb = cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB)
            cropped_box = box_rgb[y1:y2, x1:x2]
            pil_box = Image.fromarray(cropped_box)
            #保存
            extracted_masks.append(pil_mask.convert("RGB"))
            extracted_boxes.append(pil_box.convert("RGB"))
        #从里面选取CLIP分数最高的
        maxscore=-1.0
        target_mask=None
        target_box=None
        for i in range(len(extracted_boxes)):
            try:
                mask,box=extracted_masks[i],extracted_boxes[i]
                score=max(CLIPScore(mask,text_prompt),CLIPScore(box,text_prompt))
                if score>maxscore:
                    maxscore=score
                    target_box=box
                    target_mask=mask
            except Exception as e:
                Debug("Exception:",e)
        if maxscore<0.0:
            raise Exception("None capture")
        return target_mask,target_box
    #
    def EnsureGet(text_threshold,box_threshold):
        if text_threshold<0.0 or box_threshold<0.0:
            return None,None
        try:
            return run(text_threshold,box_threshold)
        except Exception as e:
            return EnsureGet(text_threshold-0.05,box_threshold-0.05)
    return EnsureGet(0.8,0.8)
