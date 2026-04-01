import json
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glob import glob
import cv2
from collections import defaultdict  # 新增：用于总统计的字典

# 中文到英文类别映射（保持与训练集一致）
category_mapping = {
    "血管内皮细胞": 7,  # Vascular Endothelial Cell (VEC)
    "纤维间质": 8,  # Stromal Cell (STR)
    "嗜酸性粒细胞": 4,  # Eosinophil (EOI)
    "Tumor": 1,  # Tumor Cell (TUM)
    "淋巴细胞": 2,  # Lymphocyte (LYM)
    "中性粒细胞": 3,  # Neutrophil (NEU)
    "嗜酸性粒细胞核": 5,  # Eosinophil Nucleus (EON)
    "浆细胞": 6,  # Plasma Cell (PLS)
    "组织细胞": 9  # Histiocyte (HIS)
}

# 反向映射：ID到中文名称（新增：用于总统计）
id_to_chinese = {
    1: "Tumor",
    2: "淋巴细胞",
    3: "中性粒细胞",
    4: "嗜酸性粒细胞",
    5: "嗜酸性粒细胞核",
    6: "浆细胞",
    7: "血管内皮细胞",
    8: "纤维间质",
    9: "组织细胞"
}

# 反向映射：ID到英文缩写（3-5个字母，用于可视化和输出）
id_to_abbr = {
    1: "TUM",  # Tumor Cell
    2: "LYM",  # Lymphocyte
    3: "NEU",  # Neutrophil
    4: "EOI",  # Eosinophil
    5: "EON",  # Eosinophil Nucleus
    6: "PLS",  # Plasma Cell
    7: "VEC",  # Vascular Endothelial Cell
    8: "STR",  # Stromal Cell
    9: "HIS"  # Histiocyte
}

# COCO数据集类别定义（保持与训练集一致）
categories = [
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

# 为每个类别定义颜色（用于mask可视化）
category_colors = {
    1: (255, 0, 0),  # Red - Tumor Cell
    2: (0, 255, 0),  # Green - Lymphocyte
    3: (0, 0, 255),  # Blue - Neutrophil
    4: (255, 255, 0),  # Yellow - Eosinophil
    5: (255, 0, 255),  # Magenta - Eosinophil Nucleus
    6: (0, 255, 255),  # Cyan - Plasma Cell
    7: (255, 165, 0),  # Orange - Vascular Endothelial Cell
    8: (128, 0, 128),  # Purple - Stromal Cell
    9: (128, 128, 128)  # Gray - Histiocyte
}

# 颜色名称映射（用于控制台输出）
color_name_map = {
    (255, 0, 0): "Red",
    (0, 255, 0): "Green",
    (0, 0, 255): "Blue",
    (255, 255, 0): "Yellow",
    (255, 0, 255): "Magenta",
    (0, 255, 255): "Cyan",
    (255, 165, 0): "Orange",
    (128, 0, 128): "Purple",
    (128, 128, 128): "Gray"
}

# mAP数据（从你提供的结果中提取）
MAP_DATA = {
    "/home/data/jy/GLIP/DATASET/20260204/原图/22-12242-282023-07-03_10_16_32_patch_x169_y57_colorR_TargetTumorArea20681_Perimeter636_vector2targetbyVX781VY271.png": 0.3273,
    "/home/data/jy/GLIP/DATASET/20260204/原图/20-06916-42023-06-30_12_52_01_patch_x133_y66_colorR_TargetTumorArea25139373_Perimeter84519_vector2targetbyVX459VY-507.png": 0.4123,
    "/home/data/jy/GLIP/DATASET/20260204/原图/22-00344-142023-07-02_13_49_58_patch_x182_y42_colorR_TargetTumorArea22177_Perimeter719_vector2targetbyVX-233VY-885.png": 0.4095,
    "/home/data/jy/GLIP/DATASET/20260204/原图/20-06916-12023-06-30_12_50_26_patch_x48_y62_colorR_TargetTumorArea75248399_Perimeter143425_vector2targetbyVX-423VY91.png": 0.4219,
    "/home/data/jy/GLIP/DATASET/20260204/原图/21-04394-42023-07-02_09_15_49_patch_x110_y85_colorR_TargetTumorArea1484390_Perimeter13286_vector2targetbyVX-382VY765.png": 0.2176,
    "/home/data/jy/GLIP/DATASET/20260204/原图/20-06113-32023-06-30_13_56_38_patch_x97_y125_colorR_TargetTumorArea114965065_Perimeter256469_vector2targetbyVX17VY-26.png": 0.4259,
    "/home/data/jy/GLIP/DATASET/20260204/原图/22-09379-322023-07-03_08_36_40_patch_x137_y115_colorR_TargetTumorArea0_Perimeter0_vector2targetbyVX0VY0.png": 0.2841,
    "/home/data/jy/GLIP/DATASET/20260204/原图/21-08968-162023-07-02_12_11_47_patch_x95_y144_colorR_TargetTumorArea2008320_Perimeter13794_vector2targetbyVX284VY249.png": 0.3263,
    "/home/data/jy/GLIP/DATASET/20260204/原图/22-00344-22023-07-02_13_34_18_patch_x86_y100_colorR_TargetTumorArea118102_Perimeter1559_vector2targetbyVX1158VY-34.png": 0.2602,
    "/home/data/jy/GLIP/DATASET/20260204/原图/21-04268-182023-07-02_09_10_04_patch_x97_y148_colorR_TargetTumorArea23136_Perimeter734_vector2targetbyVX531VY-998.png": 0.3563,
    "/home/data/jy/GLIP/DATASET/20260204/原图/21-04793-32023-07-02_09_26_19_patch_x69_y15_colorR_TargetTumorArea109531450_Perimeter249522_vector2targetbyVX163VY-28.png": 0.3213,
    "/home/data/jy/GLIP/DATASET/20260204/原图/21-08968-212023-07-02_12_18_56_patch_x167_y124_colorR_TargetTumorArea2516582_Perimeter19354_vector2targetbyVX87VY-107.png": 0.2295
}


def get_font(font_size, bold=False):
    """
    获取支持英文的字体（简化版）
    :param font_size: 字体大小
    :param bold: 是否加粗
    :return: ImageFont对象
    """
    # 通用英文字体路径（覆盖Linux/Windows/Mac）
    font_paths = [
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        # Mac
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNSDisplay.ttf"
    ]

    # 尝试加载通用字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                print(f"Successfully loaded font: {font_path} (size: {font_size})")
                return font
            except Exception as e:
                continue

    # 使用默认字体
    print(f"Warning: No custom font found, using default font (size: {font_size})")
    return ImageFont.load_default()


def check_image_quality(image_path, threshold=0.3):
    """
    检查图片的mAP值是否符合要求
    :param image_path: 图片路径
    :param threshold: mAP阈值
    :return: bool: True表示符合要求，False表示不符合
    """
    # 标准化路径（处理绝对/相对路径问题）
    abs_image_path = os.path.abspath(image_path)
    # 检查是否在mAP数据中
    for map_path, map_value in MAP_DATA.items():
        if os.path.basename(map_path) == os.path.basename(image_path):
            if map_value >= threshold:
                print(f"✅ Image {os.path.basename(image_path)} mAP: {map_value} ≥ {threshold}, keep")
                return True
            else:
                print(f"❌ Image {os.path.basename(image_path)} mAP: {map_value} < {threshold}, skip")
                return False
    # 如果不在mAP数据中，默认保留
    print(f"ℹ️ Image {os.path.basename(image_path)} not in mAP data, keep by default")
    return True


def create_mask_from_polygon(polygon, width, height):
    """Create mask from polygon coordinates (enhanced coordinate compatibility)"""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    try:
        formatted_polygon = []
        for point in polygon:
            if isinstance(point, list) and len(point) == 1 and isinstance(point[0], (list, tuple)):
                point = point[0]
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = float(point[0]), float(point[1])
                formatted_polygon.append((x, y))
            else:
                print(f"Invalid coordinate format: {point}, skipped")
        if len(formatted_polygon) >= 3:
            draw.polygon(formatted_polygon, fill=255)
        else:
            print(f"Insufficient polygon points ({len(formatted_polygon)}), cannot draw")
    except (TypeError, ValueError) as e:
        print(f"Error processing polygon: {e}")
        print(f"Original coordinate data: {polygon}")
    return np.array(mask)


def extract_cell_annotations(geojson_data):
    """Extract cell annotations from GeoJSON (enhanced coordinate processing)"""
    annotations = []
    # Count category instances
    category_count = {}
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        if "classification" in props:
            geometry = feature.get("geometry", {})
            if geometry.get("type") == "Polygon":
                coordinates = geometry.get("coordinates", [])
                while len(coordinates) > 0 and isinstance(coordinates[0], list) and len(
                        coordinates[0]) > 0 and isinstance(coordinates[0][0], list):
                    coordinates = coordinates[0]
                class_info = props["classification"]
                category_name = class_info.get("name", "")
                # Count instances
                if category_name in category_count:
                    category_count[category_name] += 1
                else:
                    category_count[category_name] = 1
                annotations.append({
                    "id": feature.get("id", ""),
                    "category": category_name,
                    "coordinates": coordinates,
                    "color": class_info.get("color", [255, 255, 255])
                })
    # Print detailed statistics
    print("\n===== Category Instance Statistics (Current File) =====")
    for cat_name, count in category_count.items():
        if cat_name in category_mapping:
            cat_id = category_mapping[cat_name]
            cat_abbr = id_to_abbr.get(cat_id, "UNK")
            color_rgb = category_colors[cat_id]
            color_name = color_name_map.get(color_rgb, f"RGB{color_rgb}")
            print(
                f"Category: {cat_abbr} (ID:{cat_id}, {cat_name}) | Count: {count} | Drawing Color: {color_name} (RGB: {color_rgb})")
        else:
            print(f"Category: {cat_name} | Count: {count} | Drawing Color: Undefined (not in mapping table)")
    print("========================================\n")
    return annotations, category_count  # 修改：返回当前文件的类别统计


def visualize_annotations_on_original(image_path, annotations, output_dir):
    """Visualize annotations on original image (ONLY cell bbox + label)"""
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    draw = ImageDraw.Draw(image)

    # Load font
    font_small = get_font(8)  # Small font for cell labels

    # Draw each annotation (mask + bbox + label)
    for ann_idx, ann in enumerate(annotations):
        chinese_name = ann["category"]
        if chinese_name not in category_mapping:
            continue
        category_id = category_mapping[chinese_name]
        color = category_colors.get(category_id, (255, 255, 255))
        polygon = ann["coordinates"]
        if not polygon or not isinstance(polygon, list):
            print(f"Annotation {ann_idx} has no valid coordinates, skipped")
            continue

        # 1. Draw mask
        ann_mask = create_mask_from_polygon(polygon, width, height)
        for c in range(3):
            mask[:, :, c] = np.where(ann_mask == 255, color[c], mask[:, :, c])

        # 2. Calculate and draw bbox
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        # Draw bbox (outline width 2, same color as mask)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

        # 3. Draw label (small font, top-left, black stroke for readability)
        label_text = id_to_abbr.get(category_id, "UNK")
        # Avoid label out of image bounds
        label_x = max(0, x_min)
        label_y = max(0, y_min - 10)
        # Black stroke
        draw.text((label_x + 1, label_y), label_text, fill=(0, 0, 0), font=font_small)
        draw.text((label_x - 1, label_y), label_text, fill=(0, 0, 0), font=font_small)
        draw.text((label_x, label_y + 1), label_text, fill=(0, 0, 0), font=font_small)
        draw.text((label_x, label_y - 1), label_text, fill=(0, 0, 0), font=font_small)
        # Main text
        draw.text((label_x, label_y), label_text, fill=color, font=font_small)

    # Merge original image and mask
    combined = cv2.addWeighted(image_np, 0.7, mask, 0.3, 0)
    # Convert back to PIL Image
    combined_img = Image.fromarray(combined)
    output_filename = os.path.join(output_dir, f"original_gt_{os.path.basename(image_path)}")
    combined_img.save(output_filename)
    print(f"Original GT image saved to: {output_filename}")
    return output_filename


def visualize_patch_annotations(image_path, annotations, output_dir):
    """Visualize GT for 224x224 images (no cropping needed)"""
    os.makedirs(output_dir, exist_ok=True)

    # Load font
    font_small = get_font(8)  # Small font for cell labels

    patch_img = Image.open(image_path).convert("RGB")
    patch_np = np.array(patch_img)
    patch_w, patch_h = patch_img.size
    mask = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)

    # Process each annotation (draw mask first)
    for ann in annotations:
        chinese_name = ann["category"]
        if chinese_name not in category_mapping:
            continue
        category_id = category_mapping[chinese_name]
        color = category_colors.get(category_id, (255, 255, 255))
        coords = ann["coordinates"]

        if len(coords) >= 3:
            # 1. Draw mask
            ann_mask = create_mask_from_polygon(coords, patch_w, patch_h)
            for c in range(3):
                mask[:, :, c] = np.where(ann_mask == 255, color[c], mask[:, :, c])

    # Merge patch and mask
    combined = cv2.addWeighted(patch_np, 0.7, mask, 0.3, 0)
    # Convert back to PIL Image for drawing bbox and label
    combined_img = Image.fromarray(combined)
    draw = ImageDraw.Draw(combined_img)

    # Re-iterate annotations to draw bbox and label
    for ann in annotations:
        chinese_name = ann["category"]
        if chinese_name not in category_mapping:
            continue
        category_id = category_mapping[chinese_name]
        color = category_colors.get(category_id, (255, 255, 255))
        coords = ann["coordinates"]

        if len(coords) >= 3:
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)

            # Draw bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)

            # Draw label (small font, top-left, black stroke)
            label_text = id_to_abbr.get(category_id, "UNK")
            label_x = max(0, x_min)
            label_y = max(0, y_min - 8)
            # Black stroke
            draw.text((label_x + 1, label_y), label_text, fill=(0, 0, 0), font=font_small)
            draw.text((label_x - 1, label_y), label_text, fill=(0, 0, 0), font=font_small)
            draw.text((label_x, label_y + 1), label_text, fill=(0, 0, 0), font=font_small)
            draw.text((label_x, label_y - 1), label_text, fill=(0, 0, 0), font=font_small)
            # Main text
            draw.text((label_x, label_y), label_text, fill=color, font=font_small)

    # Save result
    output_filename = os.path.join(output_dir, f"patch_gt_{os.path.basename(image_path)}")
    combined_img.save(output_filename)
    print(f"GT image saved to: {output_filename}")
    return output_filename


def polygon_to_coco_segmentation(polygon):
    """Convert polygon coordinates to COCO-style segmentation list"""
    return [coord for point in polygon for coord in point]


def calculate_bbox(polygon):
    """Calculate bounding box from polygon coordinates"""
    if not polygon:
        return [0, 0, 0, 0]
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    x_min, y_min = min(x_coords), min(y_coords)
    x_max, y_max = max(x_coords), max(y_coords)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def calculate_polygon_area(polygon):
    """Calculate polygon area"""
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def generate_coco_annotations(image_path, annotations, global_img_id, global_ann_id):
    """Generate COCO-style annotation data for 224x224 images"""
    coco_images = []
    coco_annotations = []

    # Get image info
    img = Image.open(image_path)
    patch_w, patch_h = img.size

    # 1. Image info: use global ID
    global_img_id[0] += 1
    current_img_id = global_img_id[0]

    # Add image info (use absolute path)
    coco_images.append({
        "id": current_img_id,
        "width": patch_w,
        "height": patch_h,
        "file_name": os.path.abspath(image_path),
        "original_image": os.path.basename(image_path)
    })

    # 2. Annotation info
    for ann in annotations:
        chinese_name = ann["category"]
        if chinese_name not in category_mapping:
            continue
        category_id = category_mapping[chinese_name]
        coords = ann["coordinates"]

        if len(coords) >= 3:
            global_ann_id[0] += 1
            current_ann_id = global_ann_id[0]
            segmentation = [polygon_to_coco_segmentation(coords)]
            bbox = calculate_bbox(coords)
            area = calculate_polygon_area(coords)
            coco_annotations.append({
                "id": current_ann_id,
                "image_id": current_img_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0
            })

    return coco_images, coco_annotations


def main():
    # Parse command line arguments (适配新的20260204数据集路径)
    parser = argparse.ArgumentParser(description="Convert GeoJSON to COCO dataset (20260204 224x224 dataset)")
    parser.add_argument("--geojson_dir", default="/home/data/jy/GLIP/DATASET/20260204/",
                        help="Directory of GeoJSON files (20260204 dataset)")
    parser.add_argument("--image_dir", default="/home/data/jy/GLIP/DATASET/20260204/原图",
                        help="Directory of original 224x224 images (20260204 dataset)")
    parser.add_argument("--output_json", default="/home/data/jy/GLIP/DATASET/20260204/out/coco_annotations.json",
                        help="Output path for COCO JSON (20260204 dataset)")
    parser.add_argument("--vis_original_dir", default="/home/data/jy/GLIP/DATASET/20260204/out/visualizations",
                        help="Directory to save GT images (20260204 dataset)")
    parser.add_argument("--debug", type=bool, default=True,
                        help="Enable GT visualization")
    parser.add_argument("--selected_good_val", type=bool, default=True,
                        help="Whether to select only images with mAP ≥ 0.3 (default: True)")

    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(args.vis_original_dir, exist_ok=True)

    # Initialize total COCO data structure
    total_coco = {
        "info": {
            "description": "20260204 Dataset (224x224, mAP ≥ 0.3)" if args.selected_good_val else "20260204 Dataset (224x224)",
            "version": "1.0", "year": 2026},
        "licenses": [{"id": 1, "name": "Unknown License"}],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # Global ID counters
    global_img_id = [0]
    global_ann_id = [0]

    # 新增：初始化总类别计数器
    total_category_count = defaultdict(int)

    # 统计变量
    total_processed = 0
    total_skipped = 0
    total_kept = 0

    # Process 20260204 dataset GeoJSON files
    geojson_files = glob(os.path.join(args.geojson_dir, "*.geojson"))
    print(f"Found {len(geojson_files)} GeoJSON files in 20260204 dataset")

    # 打印过滤配置
    if args.selected_good_val:
        print(f"📝 Filter enabled: Only keep images with mAP ≥ 0.3")
    else:
        print(f"📝 Filter disabled: Keep all images")

    for geojson_path in geojson_files:
        total_processed += 1

        # Get corresponding original image path
        geojson_basename = os.path.splitext(os.path.basename(geojson_path))[0]

        # 处理图片扩展名（可能是jpg或png）
        image_path_jpg = os.path.join(args.image_dir, f"{geojson_basename}.jpg")
        image_path_png = os.path.join(args.image_dir, f"{geojson_basename}.png")

        if os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        elif os.path.exists(image_path_png):
            image_path = image_path_png
        else:
            print(f"Warning: Original image {geojson_basename}.(jpg/png) not found, skipped")
            total_skipped += 1
            continue

        # 检查mAP值（如果开启过滤）
        if args.selected_good_val and not check_image_quality(image_path):
            total_skipped += 1
            continue

        total_kept += 1

        print(f"\n========== Processing file: {geojson_basename} ==========")

        # Read GeoJSON data
        with open(geojson_path, 'r', encoding='utf-8') as f:
            try:
                geojson_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {geojson_path} has invalid format, skipped")
                total_skipped += 1
                total_kept -= 1
                continue

        # Extract cell annotations
        annotations, curr_file_cat_count = extract_cell_annotations(geojson_data)

        # 新增：累加当前文件的类别统计到总计数器
        for cat_name, count in curr_file_cat_count.items():
            if cat_name in category_mapping:
                cat_id = category_mapping[cat_name]
                total_category_count[cat_id] += count
            else:
                # 未映射的类别单独统计
                total_category_count[cat_name] += count

        print(f"Found {len(annotations)} cell annotations")

        # Generate COCO annotations (直接处理224x224图像，无需裁剪)
        coco_images, coco_annotations = generate_coco_annotations(
            image_path, annotations, global_img_id, global_ann_id
        )

        # Merge to total COCO data
        total_coco["images"].extend(coco_images)
        total_coco["annotations"].extend(coco_annotations)

        # GT visualization
        if args.debug:
            visualize_patch_annotations(
                image_path, annotations, args.vis_original_dir
            )

    # Save 20260204 dataset COCO JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(total_coco, f, ensure_ascii=False, indent=2)

    # Print processing results
    print(f"\n==================== 20260204 Dataset Processing Completed! ====================")
    print(f"COCO annotation file saved to: {args.output_json}")
    print(f"Total files processed: {total_processed}")
    print(f"Files skipped (low mAP or missing): {total_skipped}")
    print(f"Files kept: {total_kept}")
    print(f"Total {len(total_coco['images'])} images processed (ID range: 1-{global_img_id[0]})")
    print(f"Total {len(total_coco['annotations'])} annotations generated (ID range: 1-{global_ann_id[0]})")

    # Validation checks
    image_ids = [img["id"] for img in total_coco["images"]]
    if sorted(image_ids) == list(range(1, len(image_ids) + 1)):
        print("✅ Image IDs are continuous")
    else:
        print("❌ Image IDs are not continuous")

    missing_files = 0
    for img in total_coco["images"]:
        if not os.path.exists(img["file_name"]):
            missing_files += 1
            print(f"❌ Missing image file: {img['file_name']}")
    if missing_files == 0:
        print("✅ All image files exist")

    if args.debug:
        print(f"GT images saved to: {args.vis_original_dir}")

    # 新增：打印全数据集细胞类别总统计
    print("\n==================== Total Cell Category Statistics (20260204 Dataset) ====================")
    print(f"{'类别ID':<6} {'英文缩写':<6} {'中文名称':<12} {'总实例数':<8} {'绘制颜色':<10}")
    print("-" * 60)
    total_cells = 0
    # 先打印已映射的类别
    for cat_id in sorted(category_mapping.values()):
        count = total_category_count.get(cat_id, 0)
        total_cells += count
        cat_abbr = id_to_abbr.get(cat_id, "UNK")
        cat_chinese = id_to_chinese.get(cat_id, "未知")
        color_rgb = category_colors.get(cat_id, (0, 0, 0))
        color_name = color_name_map.get(color_rgb, f"RGB{color_rgb}")
        print(f"{cat_id:<6} {cat_abbr:<6} {cat_chinese:<12} {count:<8} {color_name:<10}")
    # 打印未映射的类别（如果有）
    unmapped_cats = [k for k in total_category_count.keys() if k not in category_mapping.values()]
    if unmapped_cats:
        print("\n未映射的类别：")
        for cat_name in unmapped_cats:
            count = total_category_count[cat_name]
            total_cells += count
            print(f"{'--':<6} {'UNK':<6} {cat_name:<12} {count:<8} {'Undefined':<10}")
    print("-" * 60)
    print(f"{'总计':<6} {'--':<6} {'所有细胞':<12} {total_cells:<8} {'--':<10}")

    # 打印过滤结果汇总
    if args.selected_good_val:
        print("\n==================== mAP Filter Results ====================")
        print("Images skipped due to low mAP (<0.3):")
        for img_path, map_val in MAP_DATA.items():
            if map_val < 0.3:
                print(f"  - {os.path.basename(img_path)} (mAP: {map_val})")
        print("\nImages kept (mAP ≥ 0.3):")
        for img_path, map_val in MAP_DATA.items():
            if map_val >= 0.3:
                print(f"  - {os.path.basename(img_path)} (mAP: {map_val})")

    print("====================================================================================")


if __name__ == "__main__":
    main()