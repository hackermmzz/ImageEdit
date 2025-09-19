import threading
import time
from concurrent.futures import ThreadPoolExecutor

def print_thread_id():
    # 获取当前线程对象
    current_thread = threading.current_thread()
    # 获取线程 ID（ident 属性）
    thread_id = current_thread.ident
    # 获取线程名称（可选）
    thread_name = current_thread.name
    print(f"线程名称: {thread_name}, 线程 ID: {thread_id}")

# 主线程中获取 ID
print("主线程:")
print_thread_id()

# 子线程中获取 ID
t1 = threading.Thread(target=print_thread_id, name="子线程1")
t2 = threading.Thread(target=print_thread_id, name="子线程2")

t1.start()
t2.start()
t1.join()
t2.join()