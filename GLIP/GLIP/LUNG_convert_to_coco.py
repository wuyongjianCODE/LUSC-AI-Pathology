import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import shutil
from tqdm import tqdm
import argparse

# 类别映射关系：中文到英文及ID，包含处理特殊情况
CLASS_MAPPING = {
    "肿瘤细胞": "Tumor Cell",
    "淋巴细胞": "Lymphocyte",
    "中性粒细胞": "Neutrophil",
    "嗜酸性粒细胞": "Eosinophil",
    "嗜酸性粒细胞核": "Eosinophil Nucleus",
    "浆细胞": "Plasma Cell",
    "血管内皮细胞": "Vascular Endothelial Cell",
    "血管内皮": "Vascular Endothelial Cell",  # 将"血管内皮"视为"血管内皮细胞"
    "基质细胞": "Stromal Cell",
    "纤维间质细胞": "Stromal Cell",  # 将"纤维间质细胞"视为"间质细胞"
    "组织细胞": "Histiocyte"
}

# COCO类别定义（按指定顺序）
COCO_CATEGORIES = [
    {"id": 1, "name": "Tumor Cell", "supercategory": "cell"},
    {"id": 2, "name": "Lymphocyte", "supercategory": "cell"},
    {"id": 3, "name": "Neutrophil", "supercategory": "cell"},
    {"id": 4, "name": "Eosinophil", "supercategory": "cell"},
    {"id": 5, "name": "Eosinophil Nucleus", "supercategory": "cell"},
    {"id": 6, "name": "Plasma Cell", "supercategory": "cell"},
    {"id": 7, "name": "Vascular Endothelial Cell", "supercategory": "cell"},
    {"id": 8, "name": "Stromal Cell", "supercategory": "cell"},
    {"id": 9, "name": "Histiocyte", "supercategory": "cell"}
]


# 获取类别ID
def get_category_id(class_name):
    for cat in COCO_CATEGORIES:
        if cat["name"] == class_name:
            return cat["id"]
    return None


# 切割图像为224x224的patch
def split_image_into_patches(image_path, output_dir, patch_size=224, overlap=0):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    img_height, img_width = img.shape[:2]
    patches = []
    patch_info = []

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 计算步长
    step = patch_size - overlap

    # 生成所有patch
    for y in range(0, img_height, step):
        for x in range(0, img_width, step):
            # 计算patch的坐标
            y2 = min(y + patch_size, img_height)
            x2 = min(x + patch_size, img_width)

            # 如果patch小于指定大小，调整起始坐标以确保大小
            if y2 - y < patch_size:
                y = max(0, y2 - patch_size)
            if x2 - x < patch_size:
                x = max(0, x2 - patch_size)

            # 提取patch
            patch = img[y:y + patch_size, x:x + patch_size]

            # 生成patch文件名
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            patch_filename = f"{img_name}_patch_{x}_{y}.jpg"
            patch_path = os.path.join(output_dir, patch_filename)

            # 保存patch
            cv2.imwrite(patch_path, patch)

            patches.append(patch)
            patch_info.append({
                "path": patch_path,
                "filename": patch_filename,
                "x": x,
                "y": y,
                "width": patch_size,
                "height": patch_size,
                "original_width": img_width,
                "original_height": img_height
            })

    return patch_info


# 检查多边形是否在patch内
def is_polygon_in_patch(polygon, patch_x, patch_y, patch_width, patch_height):
    for (x, y) in polygon:
        if (patch_x <= x < patch_x + patch_width and
                patch_y <= y < patch_y + patch_height):
            return True
    return False


# 转换多边形坐标到patch坐标系
def convert_polygon_to_patch_coords(polygon, patch_x, patch_y):
    return [(x - patch_x, y - patch_y) for (x, y) in polygon]


# 计算边界框
def calculate_bbox(polygon):
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


# 转换多边形为COCO格式的分割掩码
def polygon_to_coco_segmentation(polygon):
    # COCO格式要求的是展平的列表 [x1,y1,x2,y2,...]
    return [coord for point in polygon for coord in point]


# 使用matplotlib在控制台显示标注结果
def visualize_with_plt(image_path, annotations, title="标注可视化"):
    # 读取图像并转换颜色通道（OpenCV默认BGR，matplotlib需要RGB）
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建画布
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()

    for ann in annotations:
        # 绘制边界框
        bbox = ann["bbox"]
        rect = plt.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2], bbox[3],
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(rect)

        # 绘制分割掩码
        segmentation = ann["segmentation"]
        if segmentation and len(segmentation) > 0:
            seg_flat = segmentation[0] if isinstance(segmentation[0], list) else segmentation

            if len(seg_flat) >= 2 and len(seg_flat) % 2 == 0:
                # 转换为多边形坐标
                x = seg_flat[::2]  # 偶数索引（0,2,4...）为x坐标
                y = seg_flat[1::2]  # 奇数索引（1,3,5...）为y坐标

                # 闭合多边形
                x.append(x[0])
                y.append(y[0])

                # 绘制多边形
                plt.plot(x, y, 'b-', linewidth=2)

        # 绘制类别名称
        category_id = ann["category_id"]
        category_name = next(cat["name"] for cat in COCO_CATEGORIES if cat["id"] == category_id)
        plt.text(
            bbox[0], bbox[1] - 5,
            category_name,
            color='red',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7)
        )

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 保存可视化标注到文件
def save_visualization(image_path, annotations, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for ann in annotations:
        # 绘制边界框
        bbox = ann["bbox"]
        draw.rectangle([
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3])
        ], outline="red", width=2)

        # 绘制分割掩码
        segmentation = ann["segmentation"]
        if segmentation and len(segmentation) > 0:
            seg_flat = segmentation[0] if isinstance(segmentation[0], list) else segmentation

            if len(seg_flat) >= 2 and len(seg_flat) % 2 == 0:
                points = [(seg_flat[i], seg_flat[i + 1])
                          for i in range(0, len(seg_flat), 2)]
                draw.polygon(points, outline="blue", width=2)

        # 绘制类别名称
        category_id = ann["category_id"]
        category_name = next(cat["name"] for cat in COCO_CATEGORIES if cat["id"] == category_id)
        draw.text((bbox[0], bbox[1] - 15), category_name, fill="red")

    img.save(output_path)


# 主函数：转换GeoJSON到COCO格式
def convert_geojson_to_coco(geojson_dir, images_dir, output_dir, split_ratio=0.8, debug=True):
    # 创建输出目录结构
    coco_train_dir = os.path.join(output_dir, "train")
    coco_val_dir = os.path.join(output_dir, "val")
    coco_train_images = os.path.join(coco_train_dir, "images")
    coco_val_images = os.path.join(coco_val_dir, "images")
    coco_train_gt = os.path.join(coco_train_dir, "gt_visualization")
    coco_val_gt = os.path.join(coco_val_dir, "gt_visualization")

    os.makedirs(coco_train_images, exist_ok=True)
    os.makedirs(coco_val_images, exist_ok=True)
    os.makedirs(coco_train_gt, exist_ok=True)
    os.makedirs(coco_val_gt, exist_ok=True)

    # 初始化COCO格式数据
    coco_train = {
        "info": {"description": "Medical Cell Dataset (COCO format)"},
        "licenses": [],
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": []
    }

    coco_val = {
        "info": {"description": "Medical Cell Dataset (COCO format)"},
        "licenses": [],
        "categories": COCO_CATEGORIES,
        "images": [],
        "annotations": []
    }

    # 用于收集所有被归类为间质细胞的原始中文名称
    stromal_cell_original_names = set()

    image_id = 1
    annotation_id = 1

    # 获取所有GeoJSON文件
    geojson_files = [f for f in os.listdir(geojson_dir) if f.endswith(".geojson")]

    for geojson_file in tqdm(geojson_files, desc="处理文件"):
        # 读取GeoJSON文件
        geojson_path = os.path.join(geojson_dir, geojson_file)
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # 找到对应的图像文件
        img_basename = os.path.splitext(geojson_file)[0] + ".jpg"
        img_path = os.path.join(images_dir, img_basename)

        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在 - {img_path}")
            continue

        # 切割图像为patch
        temp_patch_dir = os.path.join(output_dir, "temp_patches")
        patch_info_list = split_image_into_patches(img_path, temp_patch_dir)

        # 处理每个patch
        for idx, patch_info in enumerate(patch_info_list):
            # 决定是训练集还是验证集
            is_train = (idx % 100) < (split_ratio * 100)
            target_dir = coco_train_images if is_train else coco_val_images
            target_gt_dir = coco_train_gt if is_train else coco_val_gt
            target_coco = coco_train if is_train else coco_val

            # 复制patch到目标目录
            dest_path = os.path.join(target_dir, patch_info["filename"])
            shutil.copy(patch_info["path"], dest_path)

            # 添加图像信息到COCO
            img_info = {
                "id": image_id,
                "file_name": patch_info["filename"],
                "width": patch_info["width"],
                "height": patch_info["height"],
                "original_file": img_basename,
                "original_x": patch_info["x"],
                "original_y": patch_info["y"]
            }
            target_coco["images"].append(img_info)

            # 收集该patch的所有标注
            patch_annotations = []

            # 处理每个特征（标注）
            for feature in geojson_data.get("features", []):
                # 获取类别信息
                class_name_cn = feature["properties"]["classification"]["name"]
                if class_name_cn not in CLASS_MAPPING:
                    print(f"警告: 未知类别 - {class_name_cn}")
                    continue

                class_name_en = CLASS_MAPPING[class_name_cn]
                category_id = get_category_id(class_name_en)
                if category_id is None:
                    print(f"警告: 找不到类别ID - {class_name_en}")
                    continue

                # 检查是否为间质细胞，如果是则记录原始中文名称
                if class_name_en == "Stromal Cell":
                    stromal_cell_original_names.add(class_name_cn)

                # 获取多边形坐标
                if feature["geometry"]["type"] == "Polygon":
                    # GeoJSON的coordinates格式是[[[x1,y1], [x2,y2], ...]]
                    polygons = feature["geometry"]["coordinates"]
                    for polygon in polygons:
                        # 检查多边形是否在当前patch内
                        if is_polygon_in_patch(
                                polygon,
                                patch_info["x"],
                                patch_info["y"],
                                patch_info["width"],
                                patch_info["height"]
                        ):
                            # 转换坐标到patch坐标系
                            patch_polygon = convert_polygon_to_patch_coords(
                                polygon,
                                patch_info["x"],
                                patch_info["y"]
                            )

                            # 计算边界框
                            bbox = calculate_bbox(patch_polygon)

                            # 转换为COCO分割格式
                            segmentation = polygon_to_coco_segmentation(patch_polygon)

                            # 计算面积
                            area = bbox[2] * bbox[3]  # 简化计算

                            # 创建标注信息
                            annotation = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "bbox": bbox,
                                "area": area,
                                "segmentation": [segmentation],
                                "iscrowd": 0
                            }

                            target_coco["annotations"].append(annotation)
                            patch_annotations.append(annotation)
                            annotation_id += 1

            # 保存可视化结果
            visualize_path = os.path.join(target_gt_dir,
                                          os.path.splitext(patch_info["filename"])[0] + "_gt.jpg")
            save_visualization(dest_path, patch_annotations, visualize_path)

            # 如果开启debug模式，在控制台显示标注结果
            if debug and patch_annotations:  # 只显示有标注的patch
                visualize_with_plt(
                    dest_path,
                    patch_annotations,
                    title=f"{patch_info['filename']} - 标注预览"
                )

            image_id += 1

        # 清理临时patch目录
        if os.path.exists(temp_patch_dir):
            shutil.rmtree(temp_patch_dir)

    # 保存COCO格式的JSON文件
    with open(os.path.join(coco_train_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(coco_train, f, ensure_ascii=False, indent=2)

    with open(os.path.join(coco_val_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(coco_val, f, ensure_ascii=False, indent=2)

    # 打印所有被归类为间质细胞的原始中文名称
    print("\n=== 被归类为间质细胞(Stromal Cell)的原始中文名称 ===")
    for name in sorted(stromal_cell_original_names):
        print(f"- {name}")
    print("==============================================")

    print(f"\nCOCO数据集生成完成！输出目录: {output_dir}")
    print(f"训练集图像数: {len(coco_train['images'])}")
    print(f"训练集标注数: {len(coco_train['annotations'])}")
    print(f"验证集图像数: {len(coco_val['images'])}")
    print(f"验证集标注数: {len(coco_val['annotations'])}")


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='将GeoJSON标注转换为COCO格式数据集')
    parser.add_argument('--geojson-dir', type=str,
                        default="/home/data/jy/GLIP/DATASET/20250912QuPath标注/20250912",
                        help='GeoJSON标注文件目录')
    parser.add_argument('--images-dir', type=str,
                        default="/home/data/jy/GLIP/DATASET/20250912QuPath标注/原图",
                        help='原始图像目录')
    parser.add_argument('--output-dir', type=str,
                        default="/home/data/jy/GLIP/DATASET/20250912QuPath",
                        help='输出COCO数据集目录')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='训练集与验证集的划分比例')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='是否开启debug模式，在控制台显示标注结果')

    args = parser.parse_args()

    # 转换数据集
    convert_geojson_to_coco(
        geojson_dir=args.geojson_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        debug=args.debug
    )
