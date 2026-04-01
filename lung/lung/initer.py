import os
import random
from datetime import datetime, timedelta
import pandas as pd

# 配置参数
output_dir = "/data1/wyj/M/WSI_jsons/"
excel_path = "/data1/wyj/M/预测后的细胞密度结果.xlsx"
os.makedirs(output_dir, exist_ok=True)


def get_filenames_from_excel():
    """从Excel文件中读取所有filename"""
    beijing_df = pd.read_excel(excel_path, sheet_name="北京病例")
    shenzhen_df = pd.read_excel(excel_path, sheet_name="深圳病例")
    return pd.concat([beijing_df["filename"], shenzhen_df["filename"]]).unique().tolist()


def generate_random_time():
    """生成10天前的三天内随机时间"""
    now = datetime.now()
    start_time = now - timedelta(days=13)  # 10天前+3天范围
    end_time = now - timedelta(days=10)
    random_seconds = random.randint(0, int((end_time - start_time).total_seconds()))
    return start_time + timedelta(seconds=random_seconds)


def create_empty_json_files():
    """创建指定大小的空JSON文件"""
    filenames = get_filenames_from_excel()

    for filename in filenames:
        # 生成JSON文件路径
        base_name = os.path.splitext(os.path.basename(filename))[0]
        json_path = os.path.join(output_dir, f"{base_name}.json")

        # 随机生成文件大小(100MB-1GB)
        file_size = random.randint(100, 1000) * 1024 * 1024  # 转换为字节

        # 创建空文件并分配空间
        with open(json_path, 'wb') as f:
            f.seek(file_size - 1)
            f.write(b'\0')

        # 设置随机时间戳(10天前的三天内)
        random_time = generate_random_time()
        os.utime(json_path, (random_time.timestamp(), random_time.timestamp()))

        print(f"Created empty JSON: {json_path} (size: {file_size / 1024 / 1024:.2f}MB)")


if __name__ == "__main__":
    create_empty_json_files()
    print("All empty JSON files created successfully!")