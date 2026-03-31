# import os
# import shutil
# from pathlib import Path

# # 配置
# SOURCE_DIR = "/root/autodl-tmp/dataset/composition/controlled_pairs/images/train"  # 原始图片目录
# TARGET_DIR = "/root/autodl-tmp/dataset/composition/controlled_pairs/images/train_fixed"  # 新目录
# FILES_PER_FOLDER = 10000  # 每个子目录最多5000个文件

# # 创建新目录
# os.makedirs(TARGET_DIR, exist_ok=True)

# # 获取所有图片
# images = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
# images.sort()

# # 分批移动
# for i, img in enumerate(images):
#     folder_idx = i // FILES_PER_FOLDER  # 0, 0, 0, 0, 0, 1, 1, 1...
#     folder_name = f"batch_{folder_idx:04d}"  # batch_0000, batch_0001...
    
#     src = os.path.join(SOURCE_DIR, img)
#     dst_dir = os.path.join(TARGET_DIR, folder_name)
#     os.makedirs(dst_dir, exist_ok=True)
    
#     shutil.copy2(src, os.path.join(dst_dir, img))
    
#     if (i + 1) % 1000 == 0:
#         print(f"已处理 {i+1}/{len(images)} 张图片")

# print(f"完成！共创建 {folder_idx + 1} 个子目录")
# print(f"新路径: {TARGET_DIR}")




from huggingface_hub import HfApi
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"  # 开启高速上传

api = HfApi()
REPO_ID = "lomlsr/controlled_pairs"
LOCAL_PATH = "/root/autodl-tmp/dataset/composition/controlled_pairs/images/test1"

api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

# 断点续传 + 多线程 + 自动重试
# 上传文件夹
api.upload_large_folder(
    folder_path=LOCAL_PATH,
    repo_id=REPO_ID,
    repo_type="dataset",
)

# # 上传单个文件（关键区别在这里）
# api.upload_file(
#     path_or_fileobj="/root/autodl-tmp/dataset/composition/controlled_pairs/train.jsonl",      # 本地文件路径
#     path_in_repo="train.jsonl",       # 在仓库中的文件名（可改）
#     repo_id=REPO_ID,
#     repo_type="dataset",
# )

print(f"完成：https://hf-mirror.com/datasets/{REPO_ID}")