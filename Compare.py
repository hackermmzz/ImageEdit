import os
import ast
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 全局变量用于跟踪浏览状态
current_index = 0
image_pairs = []  # 存储所有要比较的图片对

def display_two_images(img1, img2, title, fig=None, axes=None):
    """显示两张图片，并设置点击事件"""
    # 如果没有提供图形和轴，则创建新的
    if fig is None or axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        ax1, ax2 = axes
    
    # 清除之前的图像
    ax1.clear()
    ax2.clear()
    
    # 设置整体标题
    fig.suptitle(title, fontsize=16)
    
    # 显示第一张图片
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('左侧: 点击查看上一张')
    
    # 显示第二张图片
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('右侧: 点击查看下一张')
    
    # 调整布局
    plt.tight_layout()
    
    # 设置点击事件
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    return fig, (ax1, ax2)

def on_click(event):
    """处理鼠标点击事件"""
    global current_index, image_pairs
    
    # 检查是否点击了左侧图片
    if event.inaxes == plt.gcf().axes[0]:
        # 上一张
        current_index = (current_index - 1) % len(image_pairs)
    # 检查是否点击了右侧图片
    elif event.inaxes == plt.gcf().axes[1]:
        # 下一张
        current_index = (current_index + 1) % len(image_pairs)
    else:
        return  # 点击了其他区域，不做处理
    
    # 更新显示
    update_display()

def update_display():
    """更新显示的图片对"""
    global current_index, image_pairs, fig, axes
    
    if not image_pairs:
        return
    
    img1, img2, title = image_pairs[current_index]
    fig, axes = display_two_images(img1, img2, title, fig, axes)
    plt.draw()

def compare(origin, edited_image, ins: str):
    """比较两张图片并返回结果（这里简化实现）"""
    return f"比较结果: {ins}"

def GetMyData():
    path = r"C:/Users/mmzz/Desktop/debug1"
    ret = {}
    for folder in os.listdir(path):
        pp = os.path.join(path, folder, "Total")
        if not os.path.isdir(pp):
            continue
        data = []
        for r in os.listdir(pp):
            ppp = os.path.join(pp, r)
            try:
                data.append(Image.open(ppp).convert("RGB"))
            except Exception as e:
                print(f"无法打开图片 {ppp}: {e}")
        tasks_path = os.path.join(path, folder, "tasks.txt")
        if os.path.exists(tasks_path):
            with open(tasks_path, "r", encoding="utf-8") as f:
                try:
                    tasks = ast.literal_eval(f.read())
                    data1 = [x for x, y in tasks]
                    ret[folder] = [data, data1]
                except Exception as e:
                    print(f"解析任务文件 {tasks_path} 失败: {e}")
                    ret[folder] = [data, []]
        else:
            ret[folder] = [data, []]
    return ret

def GetVincieData():
    path = r"C:/Users/mmzz/Desktop/debug"
    ret = {}
    for folder in os.listdir(path):
        pp = os.path.join(path, folder)
        if not os.path.isdir(pp):
            continue
        data = []
        for r in os.listdir(pp):
            ppp = os.path.join(pp, r)
            try:
                data.append(Image.open(ppp).convert("RGB"))
            except Exception as e:
                print(f"无法打开图片 {ppp}: {e}")
        ret[folder] = [data]
    return ret

# 获取数据并准备图片对
data0 = GetMyData()
data1 = GetVincieData()

# 准备要显示的图片对（根据实际数据结构调整）
# 这里假设两个数据源中的文件夹名称是对应的
for folder in data0:
    if folder in data1 and len(data0[folder][0]) > 0 and len(data1[folder][0]) > 0:
        for i in range(len(data0[folder][0])):
            # 取每个文件夹中的第一张图片进行配对
            my_image = data0[folder][0][i]
            vincie_image = data1[folder][0][i]
            # 使用任务描述作为标题
            title = folder+":"+data0[folder][1][i]
            image_pairs.append((my_image, vincie_image, title))

# 初始化显示
fig, axes = None, None
if image_pairs:
    print("图片浏览器已启动:")
    print("- 点击左侧图片: 查看上一张")
    print("- 点击右侧图片: 查看下一张")
    print("- 关闭窗口: 退出程序")
    update_display()
    plt.show()
else:
    print("没有找到可比较的图片对")
    