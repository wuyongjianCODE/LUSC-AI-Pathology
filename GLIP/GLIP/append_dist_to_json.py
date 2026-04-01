import json
import os
import re
import pandas as pd
import cv2
import numpy as np
from collections import defaultdict


def extract_wsi_info(patch_filename):
    """从patch文件名中提取WSI名称、patch坐标和距离矢量"""
    # 提取WSI名称 (到_patch之前的部分)
    wsi_match = re.match(r'(.*?)_patch', patch_filename)
    wsi_name = wsi_match.group(1) if wsi_match else "unknown_wsi"

    # 提取patch坐标
    coord_match = re.search(r'patch_x(\d+)_y(\d+)', patch_filename)
    x, y = 0, 0
    if coord_match:
        x = int(coord_match.group(1))
        y = int(coord_match.group(2))

    # 提取距离矢量
    vector_match = re.search(r'vector2targetbyV(\d+)VY(-?\d+)', patch_filename)
    vx, vy = 0, 0
    if vector_match:
        vx = int(vector_match.group(1))
        vy = int(vector_match.group(2))

    return wsi_name, x, y, vx, vy


def calculate_distance_vector(bbox, patch_x, patch_y, vector_vx, vector_vy, patch_width=512, patch_height=512):
    """
    计算标注到肿瘤边缘的距离矢量
    假设patch的中心是patch坐标的中心
    """
    # 计算patch中心坐标
    patch_center_x = patch_x + patch_width / 2
    patch_center_y = patch_y + patch_height / 2

    # 计算标注在patch中的中心坐标
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    bbox_center_x = bbox_x + bbox_w / 2
    bbox_center_y = bbox_y + bbox_h / 2

    # 计算标注在整个WSI中的坐标
    annotation_global_x = patch_x + bbox_center_x
    annotation_global_y = patch_y + bbox_center_y

    # 计算肿瘤边缘点坐标 (patch中心 + 距离矢量)
    tumor_edge_x = patch_center_x + vector_vx
    tumor_edge_y = patch_center_y + vector_vy

    # 计算标注到肿瘤边缘的距离矢量
    vx = tumor_edge_x - annotation_global_x
    vy = tumor_edge_y - annotation_global_y

    # 计算距离长度
    distance = np.sqrt(vx ** 2 + vy ** 2)

    return vx, vy, distance


def get_average_color(image_path, bbox):
    """计算bbox区域的平均颜色(RGB)"""
    if not os.path.exists(image_path):
        return (0, 0, 0)

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return (0, 0, 0)

    # 提取bbox区域
    x, y, w, h = map(int, bbox)
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    if w <= 0 or h <= 0:
        return (0, 0, 0)

    roi = img[y:y + h, x:x + w]

    # 转换为RGB并计算平均值
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(roi_rgb, axis=(0, 1))

    return tuple(map(int, avg_color))


def process_coco_json(json_path, image_dir, output_dir):
    """处理COCO格式的JSON文件并生成Excel表格"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 按WSI分组存储标注信息
    wsi_data = defaultdict(list)

    # 处理每个标注
    for ann in coco_data.get('annotations', []):
        # 获取图像信息
        img_id = ann['image_id']
        image_info = next((img for img in coco_data['images'] if img['id'] == img_id), None)

        if not image_info:
            continue

        image_filename = image_info['file_name']
        image_path = os.path.join(image_dir, image_filename)

        # 从文件名提取WSI信息
        wsi_name, patch_x, patch_y, vector_vx, vector_vy = extract_wsi_info(image_filename)

        # 处理类别 - 将肿瘤细胞(id=1)和浆细胞(id=6)合并为间质细胞(id=8)
        original_cat_id = ann['category_id']
        if original_cat_id in [1, 6]:  # 肿瘤细胞或浆细胞
            converted_cat_id = 8  # 间质细胞
        else:
            converted_cat_id = original_cat_id

        # 获取类别名称
        cat_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == converted_cat_id), "unknown")

        # 计算标注到肿瘤边缘的距离矢量
        bbox = ann['bbox']  # [x, y, width, height]
        vx, vy, distance = calculate_distance_vector(bbox, patch_x, patch_y, vector_vx, vector_vy)

        # 获取bbox的长径和短径
        bbox_width, bbox_height = bbox[2], bbox[3]
        major_axis = max(bbox_width, bbox_height)
        minor_axis = min(bbox_width, bbox_height)

        # 获取平均颜色
        avg_r, avg_g, avg_b = get_average_color(image_path, bbox)

        # 收集数据
        data_row = {
            'area': ann['area'],
            '类别': cat_name,
            'Vx': vx,
            'Vy': vy,
            '距离长度': distance,
            '长径': major_axis,
            '短径': minor_axis,
            '平均R值': avg_r,
            '平均G值': avg_g,
            '平均B值': avg_b,
            'patch坐标': f"x{patch_x}_y{patch_y}",
            'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        }

        wsi_data[wsi_name].append(data_row)

    # 为每个WSI创建Excel文件
    for wsi_name, annotations in wsi_data.items():
        # 创建DataFrame
        df = pd.DataFrame(annotations)

        # 调整列顺序
        columns_order = [
            'area', '类别', 'Vx', 'Vy', '距离长度',
            '长径', '短径', '平均R值', '平均G值', '平均B值',
            'patch坐标', 'bbox'
        ]
        df = df.reindex(columns=columns_order)

        # 保存为Excel
        output_path = os.path.join(output_dir, f"{wsi_name}.xlsx")
        df.to_excel(output_path, index=False)
        print(f"已生成: {output_path}")


if __name__ == "__main__":
    # 配置路径
    json_path = "/home/data/jy/GLIP/OUTPUTcentral_2017_val/0.00001_20250905_042642/mask.json"
    image_dir = "/home/data/jy/GLIP/DATASET/selected_patches/"
    output_dir = "./wsi_analysis_results"  # 输出目录，可根据需要修改

    # 处理数据并生成Excel
    process_coco_json(json_path, image_dir, output_dir)
    print("处理完成!")
