# ImageEdit Project

## 项目简介
这是一个基于SAM2模型的图像编辑工具集，包含自动掩码生成、图像预测、视频预测等功能模块。项目采用模块化设计，支持多种图像处理任务。

## 功能特性
1. 自动掩码生成
2. 图像预测
3. 视频预测
4. 负反馈处理
5. 数据集下载

## 安装指南
```bash
# 安装依赖
pip install -r requirements.txt

# 安装SAM2模型
cd sam2
pip install -e .
```

## 使用示例
```bash
# 运行主程序 (支持不同配置文件)
python main.py --config sam2/sam2_hiera_{b+,l,s,t}.yaml

# 图像预测 (支持多种输入格式)
python Predict.py --image_path ./data/test.jpg --output_dir ./results --file_type jpg

# 视频处理示例
python ProcessTask.py --video_path ./data/test.mp4 --task_type mask_generation
```

## 技术架构
### 模块设计
- **SAM2核心模块** (`sam2/`)
  - `mask_decoder.py`: 掩码解码器
  - `prompt_encoder.py`: 提示编码器
  - `automatic_mask_generator.py`: 自动掩码生成器
- **应用层模块**
  - `ImageEdit.py`: 图像编辑核心逻辑
  - `Predict.py`: 预测接口封装

## 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (推荐GPU环境)
- 其他依赖: 
```bash
pip install torch torchvision torchaudio
```
├── data/               # 数据存储目录
├── datasets/           # 数据集目录
├── Models.py           # 模型定义文件
├── ImageEdit.py        # 图像编辑核心模块
└── run.sh              # 启动脚本
```

## 贡献指南
1. Fork项目
2. 创建新分支
3. 提交代码更改
4. 创建Pull Request

## 许可证
MIT License
