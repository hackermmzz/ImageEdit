import os
import sys
import shutil

def replace_symlinks(directory, recursive=False):
    """
    替换目录中的所有符号链接为其指向的实际文件
    
    参数:
        directory (str): 要处理的目录路径
        recursive (bool): 是否递归处理子目录，默认为False
    """
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误: {directory} 不是一个有效的目录", file=sys.stderr)
        return
    
    # 遍历目录中的所有条目
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        
        # 如果是目录且需要递归处理，则递归调用
        if os.path.isdir(entry_path) and not os.path.islink(entry_path) and recursive:
            replace_symlinks(entry_path, recursive)
            continue
        
        # 检查是否是符号链接
        if os.path.islink(entry_path):
            try:
                # 获取链接指向的目标路径
                target_path = os.readlink(entry_path)
                
                # 处理相对路径
                if not os.path.isabs(target_path):
                    target_path = os.path.join(os.path.dirname(entry_path), target_path)
                    target_path = os.path.abspath(target_path)
                
                # 检查目标文件是否存在
                if not os.path.exists(target_path):
                    print(f"警告: 符号链接 {entry_path} 指向的目标 {target_path} 不存在，已跳过", file=sys.stderr)
                    continue
                
                # 检查目标是否是文件（不是目录）
                if not os.path.isfile(target_path):
                    print(f"警告: 符号链接 {entry_path} 指向的目标 {target_path} 不是文件，已跳过", file=sys.stderr)
                    continue
                
                # 保存原链接的权限
                link_stat = os.lstat(entry_path)
                
                # 删除符号链接
                os.unlink(entry_path)
                
                # 复制目标文件到原链接位置
                shutil.copy2(target_path, entry_path)
                
                # 恢复原链接的权限
                os.chmod(entry_path, link_stat.st_mode)
                
                print(f"已替换: {entry_path} -> {target_path}")
                
            except Exception as e:
                print(f"处理 {entry_path} 时出错: {str(e)}", file=sys.stderr)

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python replace_symlinks.py <目录路径> [--recursive]")
        print("功能: 将指定目录下的所有符号链接替换为它们指向的实际文件")
        print("选项:")
        print("  --recursive: 递归处理所有子目录中的符号链接")
        sys.exit(1)
    
    directory = os.path.abspath(sys.argv[1])
    recursive = len(sys.argv) > 2 and sys.argv[2].lower() == "--recursive"
    
    print(f"开始处理目录: {directory}")
    if recursive:
        print("将递归处理所有子目录")
    
    replace_symlinks(directory, recursive)
    print("处理完成")

if __name__ == "__main__":
    main()
    