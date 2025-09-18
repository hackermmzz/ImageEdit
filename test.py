from PIL import Image
import numpy as np


w, h = img.size
img_arr = np.zeros((h, w, 1), dtype=np.uint8)
# 遍历像素并赋值
for r in range(h):  # 先遍历高度
    for c in range(w):  # 再遍历宽度
        img_arr[r,c]=255 if mask[c,r] else 0 
# 从数组创建图像并保存
img = Image.fromarray(img_arr)
img.save("output.png")
