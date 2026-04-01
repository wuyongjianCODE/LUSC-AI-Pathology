import json
import os
from tqdm import tqdm  # 进度条，可选但推荐

# ==================== 配置参数（核心修改：超长合并类名） ====================
CONFIG = {
    # 原始7类→4类映射（ID+名称）
    "original_to_merged_id": {
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 4,
        6: 4,
        7: 4
    },
    # 合并后4类的详细配置（超长名称：融合原始子类视觉+功能特征，60词以上）
    "merged_categories": [
        {
            "id": 1,
            "name": "Irregular pale-stained ill-defined non-specific miscellaneous nucleated cells with no clear functional orientation, encompassing all unclassified cellular components with heterogeneous morphology and non-specific filling functions",
            "short_name": "Heterogeneous Non-specific Other",
            "supercategory": "other"
        },
        {
            "id": 2,
            "name": "Round dark-stained eosinophilic infiltrating inflammatory immune cells mediating tissue immune response functions, including scattered punctate activated leukocytes regulating inflammatory microenvironment homeostasis",
            "short_name": "Round Eosinophilic Inflammatory",
            "supercategory": "inflammatory"
        },
        {
            "id": 3,
            "name": "Polygonal well-demarcated pale/dark heterogeneous epithelial cells with dual barrier protection and pathological transformation functions, integrating healthy pale homogeneous epithelial cells (barrier function) and pleomorphic hyperchromatic malignant epithelial cells (abnormal proliferation)",
            "short_name": "Polygonal Heterogeneous Epithelial",
            "supercategory": "epithelial"
        },
        {
            "id": 4,
            "name": "Long/short spindle-shaped pale-stained mesenchymal-derived stromal cells undertaking connective support and contractile functions, covering tapering bipolar fibroblasts (synthesis), thick fascicular muscle cells (contraction) and flattened cobblestone endothelial cells (vascular barrier)",
            "short_name": "Spindle-shaped Mesenchymal Stromal",
            "supercategory": "spindle-shaped"
        }
    ],
    # 原始文件路径（可批量添加）
    "original_json_paths": [
        "/home/data/jy/GLIP/DATASET/coco2s/annotations/instances_val2017.json",
        "/home/data/jy/GLIP/DATASET/coco2s/annotations/instances_train2017.json"
    ]
}

# ==================== 核心转换函数（无修改） ====================
def convert_7class_to_4class(original_json_path):
    """
    将7分类CoNSeP COCO JSON转换为4分类版本
    :param original_json_path: 原始7分类JSON路径
    :return: 新4分类JSON保存路径
    """
    # 1. 读取原始JSON
    print(f"\n📥 读取原始文件: {original_json_path}")
    with open(original_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 2. 初始化新的4分类数据结构
    new_coco_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": coco_data.get("images", []),
        "annotations": [],
        "categories": CONFIG["merged_categories"]  # 替换为超长名称的4类配置
    }

    # 3. 转换标注（更新category_id）
    print("🔄 转换标注类别ID...")
    for ann in tqdm(coco_data["annotations"], desc="处理标注"):
        original_cat_id = ann["category_id"]
        # 映射到新的4类ID
        new_cat_id = CONFIG["original_to_merged_id"].get(original_cat_id)
        if new_cat_id is None:
            print(f"⚠️  无效的原始类别ID {original_cat_id}，跳过该标注")
            continue
        # 复制标注并更新category_id
        new_ann = ann.copy()
        new_ann["category_id"] = new_cat_id
        new_coco_data["annotations"].append(new_ann)

    # 4. 生成新文件路径（不覆盖原文件，后缀加_4class）
    dir_name = os.path.dirname(original_json_path)
    file_name = os.path.basename(original_json_path)
    new_file_name = file_name.replace(".json", "_4class.json")
    new_json_path = os.path.join(dir_name, new_file_name)

    # 5. 保存新JSON文件
    print(f"💾 保存4分类文件: {new_json_path}")
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_coco_data, f, ensure_ascii=False, indent=2)

    return new_json_path

# ==================== 批量转换主函数（无修改） ====================
def main():
    # 遍历所有原始文件进行转换
    for json_path in CONFIG["original_json_paths"]:
        # 检查原始文件是否存在
        if not os.path.exists(json_path):
            print(f"❌ 原始文件不存在: {json_path}，跳过")
            continue
        # 执行转换
        new_path = convert_7class_to_4class(json_path)
        print(f"✅ 转换完成: {new_path}")

    print("\n🎉 所有文件转换完成！")
    # 输出新文件列表
    print("\n📋 新生成的4分类文件列表:")
    for json_path in CONFIG["original_json_paths"]:
        dir_name = os.path.dirname(json_path)
        file_name = os.path.basename(json_path)
        new_file_name = file_name.replace(".json", "_4class.json")
        print(f"   - {os.path.join(dir_name, new_file_name)}")

# ==================== 运行脚本 ====================
if __name__ == "__main__":
    # 安装依赖（若未安装tqdm）
    # pip install tqdm
    main()