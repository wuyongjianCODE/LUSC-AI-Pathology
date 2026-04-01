import json
import os
from collections import defaultdict

# 数据集配置（统一修正为DATASET/根目录下的绝对路径）
DATASETS = {
    "lizard": {
        "img_dir": "/home/data/jy/GLIP/DATASET/Lizard_COCO_256_segm/val_images/",
        "ann_file": "/home/data/jy/GLIP/DATASET/Lizard_COCO_256_segm/val.json",
    },
    "monusac": {
        "img_dir": "/home/data/jy/GLIP/DATASET/coco1s/train2017",
        "ann_file": "/home/data/jy/GLIP/DATASET/coco1s/annotations/instances_train2017.json"
    },
    "consep": {
        "img_dir": "/home/data/jy/GLIP/DATASET/coco2s/val2017",
        "ann_file": "/home/data/jy/GLIP/DATASET/coco2s/annotations/instances_val2017.json"
    },
    "ccrcc": {
        "img_dir": "/home/data/jy/GLIP/DATASET/coco3s/val2017",
        "ann_file": "/home/data/jy/GLIP/DATASET/coco3s/annotations/instances_val2017.json",
        "is_train": True,
    }
}


def count_coco_categories(ann_file_path):
    """
    统计COCO格式标注文件中每个类别的实例数量

    Args:
        ann_file_path (str): 标注文件的绝对路径

    Returns:
        tuple: (category_id_to_name: 类别ID到名称的映射, category_count: 类别ID到数量的映射)
    """
    # 检查文件是否存在
    if not os.path.exists(ann_file_path):
        print(f"❌ 标注文件不存在: {ann_file_path}")
        print("   请检查路径是否正确，确认数据集文件存放在 /home/data/jy/GLIP/DATASET/ 目录下")
        return None, None

    # 读取标注文件（兼容不同编码）
    try:
        with open(ann_file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except UnicodeDecodeError:
        try:
            with open(ann_file_path, 'r', encoding='gbk') as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"❌ 读取标注文件失败: {ann_file_path}，错误: {e}")
            return None, None
    except Exception as e:
        print(f"❌ 读取标注文件失败: {ann_file_path}，错误: {e}")
        return None, None

    # 1. 构建类别ID到名称的映射
    category_id_to_name = {}
    for cat in coco_data.get("categories", []):
        cat_id = cat["id"]
        cat_name = cat.get("name", f"未知类别_{cat_id}")
        category_id_to_name[cat_id] = cat_name

    # 2. 统计每个类别的实例数
    category_count = defaultdict(int)
    total_anns = len(coco_data.get("annotations", []))
    if total_anns == 0:
        print(f"⚠️  {ann_file_path} 中无标注数据")
        return category_id_to_name, category_count

    for ann in coco_data.get("annotations", []):
        cat_id = ann["category_id"]
        category_count[cat_id] += 1

    return category_id_to_name, category_count


def print_dataset_statistics(dataset_name, cat_id_to_name, cat_count):
    """
    格式化打印数据集的类别统计结果
    """
    print(f"\n==================== {dataset_name.upper()} 数据集统计 ====================")
    if not cat_id_to_name:
        print("⚠️  无有效类别信息")
        return

    if not cat_count:
        print("⚠️  无有效标注实例")
        return

    # 按类别ID排序输出
    total_instances = 0
    print(f"{'类别ID':<8} {'类别名称':<30} {'实例数量':<8}")
    print("-" * 50)
    for cat_id in sorted(cat_count.keys()):
        cat_name = cat_id_to_name.get(cat_id, f"未知类别_{cat_id}")
        count = cat_count[cat_id]
        total_instances += count
        print(f"{cat_id:<8} {cat_name:<30} {count:<8}")

    print("-" * 50)
    print(f"{'总计':<8} {'所有类别':<30} {total_instances:<8}")
    print("========================================================================")


def main():
    """主函数：遍历所有数据集并统计"""
    # 先检查根目录是否存在
    root_dir = "/home/data/jy/GLIP/DATASET/"
    if not os.path.exists(root_dir):
        print(f"❌ 数据集根目录不存在: {root_dir}")
        print("   请确认数据集存放路径正确！")
        return

    print(f"📊 开始统计 DATASET 目录下的细胞类别数量...")
    print(f"   根目录: {root_dir}")

    # 遍历每个数据集统计
    for dataset_name, config in DATASETS.items():
        ann_file = config["ann_file"]
        # 统计当前数据集
        cat_id_to_name, cat_count = count_coco_categories(ann_file)
        # 打印统计结果
        print_dataset_statistics(dataset_name, cat_id_to_name, cat_count)

    print("\n✅ 所有数据集统计完成！")


if __name__ == "__main__":
    main()