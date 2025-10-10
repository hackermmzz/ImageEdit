import io
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download
import pandas as pd
from PIL import Image

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_processor.log"),
        logging.StreamHandler()
    ]
)

# 线程池配置 - 根据系统性能调整
MAX_WORKERS = min(16, os.cpu_count() * 2)

###################################### 下载数据集
def download_dataset():
    """下载MagicBrush数据集"""
    try:
        logging.info("开始下载数据集...")
        snapshot_download(
            repo_id="osunlp/MagicBrush",
            repo_type="dataset",
            cache_dir="dataset",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logging.info("数据集下载完成")
    except Exception as e:
        logging.error(f"数据集下载失败: {str(e)}", exc_info=True)
        raise

###################################### 解析单个Parquet文件
def parse_dataset(file_path: str, savedir: str):
    """解析单个Parquet文件并保存图片和指令"""
    try:
        # 读取Parquet文件
        needed_columns = ["img_id", "turn_index", "source_img", "target_img", "instruction"]
        df = pd.read_parquet(
            path=file_path,
            engine="pyarrow",
            columns=needed_columns
        )
        
        savedir_path = Path(savedir)
        savedir_path.mkdir(parents=True, exist_ok=True)
        
        # 提前创建所有需要的目录
        img_ids = df["img_id"].unique()
        for img_id in img_ids:
            (savedir_path / str(img_id)).mkdir(exist_ok=True)
        
        # 使用线程池并行处理图片保存
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for idx, row in df.iterrows():
                futures.append(executor.submit(
                    process_row, 
                    row, 
                    savedir_path
                ))
            
            # 显示进度
            for future in tqdm(as_completed(futures), total=len(futures), 
                              desc=f"处理 {os.path.basename(file_path)}"):
                try:
                    future.result()
                except Exception as e:
                    logging.warning(f"处理行时出错: {str(e)}")
    
    except Exception as e:
        logging.error(f"解析文件 {file_path} 失败: {str(e)}", exc_info=True)

def process_row(row, savedir_path):
    """处理单行数据，保存图片和指令"""
    try:
        img_id = row["img_id"]
        turn_index = row["turn_index"]
        dir_path = savedir_path / str(img_id)
        
        # 处理源图片
        source_img_bytes = row["source_img"]["bytes"]
        with Image.open(BytesIO(source_img_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dir_path / f"source_{turn_index}.png", optimize=True)
        
        # 处理目标图片
        target_img_bytes = row["target_img"]["bytes"]
        with Image.open(BytesIO(target_img_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dir_path / f"target_{turn_index}.png", optimize=True)
        
        # 保存指令
        (dir_path / f"task_{turn_index}.txt").write_text(row["instruction"], encoding="utf-8")
        
    except Exception as e:
        raise Exception(f"处理 img_id={img_id}, turn_index={turn_index} 时出错: {str(e)}")

###################################### 解析所有数据集
def parse_all_dataset(root: str, savedir: str):
    """递归解析所有Parquet文件"""
    root_path = Path(root)
    parquet_files = list(root_path.rglob("*.parquet"))  # 递归查找所有parquet文件
    
    if not parquet_files:
        logging.warning("未找到任何Parquet文件")
        return
    
    logging.info(f"找到 {len(parquet_files)} 个Parquet文件，开始处理...")
    
    for file_path in parquet_files:
        logging.info(f"开始处理: {file_path}")
        parse_dataset(str(file_path), savedir)
        logging.info(f"完成处理: {file_path}")

###################################### 初始化函数
def init():
    """初始化数据集：如果不存在则下载并解析"""
    try:
        # 检查数据集是否已下载
        if not os.path.exists("dataset/"):
            download_dataset()
        else:
            logging.info("数据集已存在，跳过下载")
        
        # 检查解析后的数据是否已存在
        output_dir = "data/MagicBrush"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            parse_all_dataset("dataset/", output_dir)
            logging.info("所有数据集解析完成")
        else:
            logging.info("解析后的数据集已存在，跳过解析")
            
    except Exception as e:
        logging.error(f"初始化失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    init()
