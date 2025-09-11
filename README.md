# 科研训练图像编辑系统

本项目是一个多模块的图像智能编辑与评估系统，集成了大模型驱动的任务分解、编辑指令优化、自动分割、区域打分、专家评测等功能，适用于科研训练和自动化图像处理任务。

## 目录结构

```
├── main.py                # 主入口，包含流程控制与数据处理
├── ImageEdit.py           # 图像编辑API，基于Qwen-Image-Edit等模型
├── GroundedSam2.py        # 基于GroundingDINO和SAM2的目标检测与分割
├── Model.py               # 任务分解、打分、场景描述等专家模块
├── Tips.py                # 提示词与评分规则
├── VLM.py                 # 多模态大模型接口
├── LLM.py                 # 文本大模型接口
├── test.py                # 测试与调试脚本
├── sam2/                  # SAM2相关源码与工具
├── checkpoints/           # 预训练模型权重
├── data/                  # 数据集与样例
├── datasets/              # 原始/处理后数据集
├── debug/                 # 调试输出
├── tmp/                   # 临时文件
└── README.md              # 项目说明
```

## 主要功能

- **自动任务分解**：输入编辑指令，自动拆分为多轮子任务。
- **场景理解**：对输入图片进行全局与局部内容分析，生成详细描述。
- **智能编辑**：调用大模型自动优化编辑指令，提升可执行性和视觉效果。
- **目标检测与分割**：结合GroundingDINO和SAM2，自动生成精准掩码。
- **区域与全局打分**：对编辑结果进行局部和全局评分，辅助优化。
- **专家评测**：多轮编辑后自动综合评估，输出最终分数与优化建议。
- **调试与测试**：支持批量数据处理与自动化测试。

## 环境依赖

- Python 3.8+
- torch、transformers、diffusers、Pillow、numpy、opencv-python
- 需下载相关预训练模型权重至 `checkpoints/` 目录

## 快速开始

1. **安装依赖**  
   推荐使用虚拟环境：
   ```sh
   pip install -r requirements.txt
   ```

2. **准备模型权重**  
   下载SAM2、GroundingDINO等权重，放入 `checkpoints/` 目录。

3. **运行主流程**  
   ```sh
   python main.py
   ```
   按提示输入图片路径和编辑指令，系统将自动完成分解、编辑、评分与保存。

4. **批量测试**  
   将数据集放入 `data/` 目录，设置 `TEST_MODE=True` 后运行 `main.py` 可自动批量处理。

## 进阶用法

- 修改 `Tips.py` 可自定义评分规则和提示词。
- 编辑 `Model.py` 可扩展专家模块和打分逻辑。
- 使用 `test.py` 进行单步调试和分割效果测试。

## 致谢

本项目部分代码参考了 [facebookresearch/SAM2](https://github.com/facebookresearch/sam2) 及相关开源模型，感谢社区贡献。

---

如有问题或建议，欢迎联系作者或提