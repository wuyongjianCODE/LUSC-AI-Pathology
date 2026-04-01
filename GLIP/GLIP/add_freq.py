import json
import os


def add_lvis_frequency_to_object365(input_json_path, output_json_path):
    """
    为 Object365 JSON 数据集添加 LVIS 风格的 frequency 字段
    frequency 取值：f（frequent，常见）、c（common，普通）、r（rare，稀有）
    划分逻辑（基于 Object365 类别分布经验）：
    - 前 60 类（1-60）：f（常见，多为日常高频物体）
    - 中间 180 类（61-240）：c（普通，中等出现频率）
    - 后 125 类（241-365）：r（稀有，低出现频率）
    """
    # 加载原始 JSON
    print(f"加载文件: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 为每个类别添加 frequency 字段
    updated_categories = []
    for cat in data["categories"]:
        cat_id = cat["id"]
        # 按类别ID划分频率（经验值，确保覆盖所有365类）
        if 1 <= cat_id <= 60:
            cat["frequency"] = "f"  # 常见
        elif 61 <= cat_id <= 240:
            cat["frequency"] = "c"  # 普通
        elif 241 <= cat_id <= 365:
            cat["frequency"] = "r"  # 稀有
        else:
            # 异常类别ID默认设为普通（避免报错）
            cat["frequency"] = "c"
        updated_categories.append(cat)

    # 更新 categories 字段
    data["categories"] = updated_categories

    # 保存新文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    freq_count = {"f": 0, "c": 0, "r": 0}
    for cat in updated_categories:
        freq_count[cat["frequency"]] += 1
    print(f"添加 frequency 完成！统计：")
    print(f"常见类别（f）: {freq_count['f']} 个")
    print(f"普通类别（c）: {freq_count['c']} 个")
    print(f"稀有类别（r）: {freq_count['r']} 个")
    print(f"输出文件已保存: {output_json_path}")


if __name__ == "__main__":
    # --------------------------
    # 请根据实际路径修改
    # --------------------------
    INPUT_JSON = "DATASET/Object365/val/zhiyuan_objv2_val.json"  # 原始 Object365 JSON 路径
    OUTPUT_JSON = "DATASET/Object365/val/zhiyuan_objv2_val_with_freq.json"  # 带 frequency 的输出路径

    # 执行添加操作
    add_lvis_frequency_to_object365(INPUT_JSON, OUTPUT_JSON)