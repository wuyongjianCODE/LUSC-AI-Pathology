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
    "纤维间质": 8,     # Stromal Cell (STR)
    "嗜酸性粒细胞": 4, # Eosinophil (EOI)
    "Tumor": 1,        # Tumor Cell (TUM)
    "淋巴细胞": 2,     # Lymphocyte (LYM)
    "中性粒细胞": 3,   # Neutrophil (NEU)
    "嗜酸性粒细胞核": 5, # Eosinophil Nucleus (EON)
    "浆细胞": 6,       # Plasma Cell (PLS)
    "组织细胞": 9      # Histiocyte (HIS)
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
    9: "HIS"   # Histiocyte
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


def extract_red_boxes(geojson_data):
    """Extract red boxes (features without classification) from GeoJSON"""
    red_boxes = []
    for feature in geojson_data.get("features", []):
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Polygon":
            coords = geometry.get("coordinates", [[]])[0]
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width = x_max - x_min
            height = y_max - y_min
            if width > 100 and height > 100:
                red_boxes.append({
                    "bbox": (x_min, y_min, width, height),
                    "coordinates": coords
                })
    return red_boxes


def crop_patches(image_path, red_boxes, patch_size=(224, 224), overlap=0.5, patch_dir="patches"):
    """Crop image into 224x224 patches based on red boxes (allow overlap)"""
    os.makedirs(patch_dir, exist_ok=True)
    image = Image.open(image_path)
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    patch_info = []
    patch_id_counter = 0  # Patch counter within single image

    # Calculate step size
    step_x = int(patch_size[0] * (1 - overlap))
    step_y = int(patch_size[1] * (1 - overlap))

    for red_box_idx, red_box in enumerate(red_boxes):
        x, y, w, h = red_box["bbox"]
        red_box_area = image.crop((x, y, x + w, y + h))
        red_box_w, red_box_h = red_box_area.size

        # Calculate number of patches needed to cover red box
        num_patches_x = max(1, (red_box_w - patch_size[0] + step_x - 1) // step_x + 1)
        num_patches_y = max(1, (red_box_h - patch_size[1] + step_y - 1) // step_y + 1)

        # Generate all patches
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                # Calculate patch top-left in red box
                patch_x_in_redbox = i * step_x
                patch_y_in_redbox = j * step_y

                # Ensure last patch doesn't exceed red box
                if patch_x_in_redbox + patch_size[0] > red_box_w:
                    patch_x_in_redbox = max(0, red_box_w - patch_size[0])
                if patch_y_in_redbox + patch_size[1] > red_box_h:
                    patch_y_in_redbox = max(0, red_box_h - patch_size[1])

                # Calculate patch top-left in original image
                patch_x_in_original = x + patch_x_in_redbox
                patch_y_in_original = y + patch_y_in_redbox

                # Crop patch
                patch = red_box_area.crop((
                    patch_x_in_redbox,
                    patch_y_in_redbox,
                    patch_x_in_redbox + patch_size[0],
                    patch_y_in_redbox + patch_size[1]
                ))

                # Save patch
                patch_filename = f"{image_basename}_redbox_{red_box_idx}_patch_{patch_id_counter}.jpg"
                patch_path = os.path.join(patch_dir, patch_filename)
                patch.save(patch_path)

                # Record patch info
                patch_info.append({
                    "patch_id": patch_id_counter,
                    "patch_path": patch_path,
                    "patch_abs_path": os.path.abspath(patch_path),
                    "patch_filename": patch_filename,
                    "original_bbox": (x, y, w, h),
                    "patch_in_original": (patch_x_in_original, patch_y_in_original,
                                          patch_size[0], patch_size[1]),
                    "patch_in_redbox": (patch_x_in_redbox, patch_y_in_redbox,
                                        patch_size[0], patch_size[1]),
                    "patch_size": patch_size,
                    "red_box_idx": red_box_idx,
                    "original_coords": red_box["coordinates"]
                })

                patch_id_counter += 1

    return patch_info


def convert_coords(original_coords, patch_info):
    """Convert original image coordinates to patch coordinates"""
    patch_x, patch_y, _, _ = patch_info["patch_in_original"]
    converted = []
    for (x, y) in original_coords:
        px = x - patch_x
        py = y - patch_y
        if 0 <= px <= patch_info["patch_size"][0] and 0 <= py <= patch_info["patch_size"][1]:
            converted.append((px, py))
    return converted


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
            print(f"Category: {cat_abbr} (ID:{cat_id}, {cat_name}) | Count: {count} | Drawing Color: {color_name} (RGB: {color_rgb})")
        else:
            print(f"Category: {cat_name} | Count: {count} | Drawing Color: Undefined (not in mapping table)")
    print("========================================\n")
    return annotations, category_count  # 修改：返回当前文件的类别统计


def visualize_annotations_on_original(image_path, red_boxes, annotations, output_dir):
    """Visualize annotations on original image (ONLY cell bbox + label, no ROI/redbox text)"""
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    draw = ImageDraw.Draw(image)

    # Load font
    font_small = get_font(8)   # Small font for cell labels

    # 移除红框绘制和ROI文字（如果不需要红框显示，保留这部分注释；如果需要红框但不要文字，删除draw.text行即可）
    # 如需完全移除红框：注释掉下面的红框绘制代码
    # for i, red_box in enumerate(red_boxes):
    #     bbox = red_box["bbox"]
    #     x, y, w, h = bbox
    #     draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
    #     # 移除ROI文字
    #     # draw.text((x, y - 15), f"ROI {i}", fill=(0, 255, 0), font=font_medium)

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


def visualize_patch_annotations(patch_info, annotations, output_dir):
    """Visualize patch GT (ONLY cell bbox + label, no patch/redbox text)"""
    os.makedirs(output_dir, exist_ok=True)
    visualization_results = []

    # Load font
    font_small = get_font(8)   # Small font for cell labels

    for patch in patch_info:
        patch_path = patch["patch_path"]
        patch_img = Image.open(patch_path).convert("RGB")
        patch_np = np.array(patch_img)
        patch_w, patch_h = patch["patch_size"]
        mask = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)

        # Process each annotation (draw mask first)
        for ann in annotations:
            chinese_name = ann["category"]
            if chinese_name not in category_mapping:
                continue
            category_id = category_mapping[chinese_name]
            color = category_colors.get(category_id, (255, 255, 255))
            original_coords = ann["coordinates"]
            patch_coords = convert_coords(original_coords, patch)

            if len(patch_coords) >= 3:
                # 1. Draw mask
                ann_mask = create_mask_from_polygon(patch_coords, patch_w, patch_h)
                for c in range(3):
                    mask[:, :, c] = np.where(ann_mask == 255, color[c], mask[:, :, c])

        # Merge patch and mask
        combined = cv2.addWeighted(patch_np, 0.7, mask, 0.3, 0)
        # Convert back to PIL Image for drawing bbox and label
        combined_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(combined_img)

        # 移除Patch/Redbox文字绘制
        # patch_pos = f"Patch {patch['patch_id']} (Redbox {patch['red_box_idx']})"
        # draw.text((10, 10), patch_pos, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0), font=font_medium)

        # Re-iterate annotations to draw bbox and label in patch
        for ann in annotations:
            chinese_name = ann["category"]
            if chinese_name not in category_mapping:
                continue
            category_id = category_mapping[chinese_name]
            color = category_colors.get(category_id, (255, 255, 255))
            original_coords = ann["coordinates"]
            patch_coords = convert_coords(original_coords, patch)

            if len(patch_coords) >= 3:
                x_coords = [p[0] for p in patch_coords]
                y_coords = [p[1] for p in patch_coords]
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

        # Save result (重命名为gt后缀，更符合GT图命名习惯)
        output_filename = os.path.join(output_dir, f"patch_gt_{os.path.basename(patch['patch_filename'])}")
        combined_img.save(output_filename)
        visualization_results.append(output_filename)
        print(f"Patch GT image saved to: {output_filename}")
    return visualization_results


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


def generate_coco_annotations(patch_info, annotations, global_img_id, global_ann_id):
    """Generate COCO-style annotation data (continuous global IDs)"""
    coco_images = []
    coco_annotations = []

    for patch in patch_info:
        # 1. Image info: use global ID
        global_img_id[0] += 1
        current_img_id = global_img_id[0]
        patch_w, patch_h = patch["patch_size"]

        # Add image info (use absolute path)
        coco_images.append({
            "id": current_img_id,
            "width": patch_w,
            "height": patch_h,
            "file_name": patch["patch_abs_path"],
            "original_image": os.path.basename(patch["patch_path"].split("_redbox")[0] + ".jpg"),
            "original_bbox": patch["original_bbox"],
            "patch_in_original": patch["patch_in_original"]
        })

        # 2. Annotation info
        for ann in annotations:
            chinese_name = ann["category"]
            if chinese_name not in category_mapping:
                continue
            category_id = category_mapping[chinese_name]
            original_coords = ann["coordinates"]
            patch_coords = convert_coords(original_coords, patch)

            if len(patch_coords) >= 3:
                global_ann_id[0] += 1
                current_ann_id = global_ann_id[0]
                segmentation = [polygon_to_coco_segmentation(patch_coords)]
                bbox = calculate_bbox(patch_coords)
                area = calculate_polygon_area(patch_coords)
                coco_annotations.append({
                    "id": current_ann_id,
                    "image_id": current_img_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": segmentation,
                    "iscrowd": 0
                })
            # else:
            #     # Empty annotation placeholder
            #     global_ann_id[0] += 1
            #     current_ann_id = global_ann_id[0]
            #     coco_annotations.append({
            #         "id": current_ann_id,
            #         "image_id": current_img_id,
            #         "category_id": 0,
            #         "bbox": [0, 0, 0, 0],
            #         "area": 0.0,
            #         "segmentation": [],
            #         "iscrowd": 1
            #     })

    return coco_images, coco_annotations


def main():
    # Parse command line arguments (适配新的2026final数据集路径)
    parser = argparse.ArgumentParser(description="Convert GeoJSON to COCO dataset (2026final full dataset)")
    parser.add_argument("--geojson_dir", default="/home/data/jy/GLIP/DATASET/lungALL-V/",
                        help="Directory of GeoJSON files (lungALL-V dataset)")
    parser.add_argument("--image_dir", default="/home/data/jy/GLIP/DATASET/lungALL-V/原图",
                        help="Directory of original images (lungALL-V dataset)")
    parser.add_argument("--output_json", default="/home/data/jy/GLIP/DATASET/lungALL-V/out/coco_annotations.json",
                        help="Output path for COCO JSON (lungALL-V dataset)")
    parser.add_argument("--patch_dir", default="/home/data/jy/GLIP/DATASET/lungALL-V/out/patches",
                        help="Directory to save patches (lungALL-V dataset)")
    parser.add_argument("--vis_original_dir", default="/home/data/jy/GLIP/DATASET/lungALL-V/out/visualizations/original",
                        help="Directory to save original GT images (lungALL-V dataset)")
    parser.add_argument("--vis_patch_dir", default="/home/data/jy/GLIP/DATASET/lungALL-V/out/visualizations/patches",
                        help="Directory to save patch GT images (lungALL-V dataset)")
    parser.add_argument("--debug", type=bool, default=True,
                        help="Enable GT visualization")
    parser.add_argument("--patch_overlap", type=float, default=0.5,
                        help="Overlap ratio between patches")

    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(args.patch_dir, exist_ok=True)
    os.makedirs(args.vis_original_dir, exist_ok=True)
    os.makedirs(args.vis_patch_dir, exist_ok=True)

    # Initialize total COCO data structure (2026final dataset description)
    total_coco = {
        "info": {"description": "LungComb 2026 Final Dataset (Full)", "version": "1.0", "year": 2026},
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

    # Process 2026final dataset GeoJSON files
    geojson_files = glob(os.path.join(args.geojson_dir, "*.geojson"))
    print(f"Found {len(geojson_files)} GeoJSON files in 2026final dataset")

    for geojson_path in geojson_files:
        # Get corresponding original image path
        geojson_basename = os.path.splitext(os.path.basename(geojson_path))[0]
        image_path = os.path.join(args.image_dir, f"{geojson_basename}.jpg")

        if not os.path.exists(image_path):
            print(f"Warning: Original image {image_path} not found, skipped")
            continue

        print(f"\n========== Processing file: {geojson_basename} ==========")

        # Read GeoJSON data
        with open(geojson_path, 'r', encoding='utf-8') as f:
            try:
                geojson_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {geojson_path} has invalid format, skipped")
                continue

        # Extract red boxes and cell annotations（修改：接收当前文件的类别统计）
        red_boxes = extract_red_boxes(geojson_data)
        annotations, curr_file_cat_count = extract_cell_annotations(geojson_data)

        # 新增：累加当前文件的类别统计到总计数器
        for cat_name, count in curr_file_cat_count.items():
            if cat_name in category_mapping:
                cat_id = category_mapping[cat_name]
                total_category_count[cat_id] += count
            else:
                # 未映射的类别单独统计
                total_category_count[cat_name] += count

        if not red_boxes:
            print(f"Warning: No red box annotations found, skipped")
            continue

        print(f"Found {len(red_boxes)} red boxes, {len(annotations)} cell annotations")

        # Crop patches
        patch_info = crop_patches(
            image_path,
            red_boxes,
            (224, 224),
            args.patch_overlap,
            args.patch_dir
        )
        print(f"Generated {len(patch_info)} 224x224 patches from red box regions")

        # Generate COCO annotations
        coco_images, coco_annotations = generate_coco_annotations(
            patch_info, annotations, global_img_id, global_ann_id
        )

        # Merge to total COCO data
        total_coco["images"].extend(coco_images)
        total_coco["annotations"].extend(coco_annotations)

        # GT visualization
        if args.debug:
            visualize_annotations_on_original(
                image_path, red_boxes, annotations, args.vis_original_dir
            )
            visualize_patch_annotations(
                patch_info, annotations, args.vis_patch_dir
            )

    # Save 2026final dataset COCO JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(total_coco, f, ensure_ascii=False, indent=2)

    # Print processing results
    print(f"\n==================== 2026final Dataset Processing Completed! ====================")
    print(f"COCO annotation file saved to: {args.output_json}")
    print(f"Total {len(total_coco['images'])} patch images generated (ID range: 1-{global_img_id[0]})")
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
        print(f"Original GT images saved to: {args.vis_original_dir}")
        print(f"Patch GT images saved to: {args.vis_patch_dir}")

    # 新增：打印全数据集细胞类别总统计
    print("\n==================== Total Cell Category Statistics (2026final Dataset) ====================")
    print(f"{'类别ID':<6} {'英文缩写':<6} {'中文名称':<12} {'总实例数':<8} {'绘制颜色':<10}")
    print("-" * 60)
    total_cells = 0
    # 先打印已映射的类别
    for cat_id in sorted(category_mapping.values()):
        count = total_category_count.get(cat_id, 0)
        total_cells += count
        cat_abbr = id_to_abbr.get(cat_id, "UNK")
        cat_chinese = id_to_chinese.get(cat_id, "未知")
        color_rgb = category_colors.get(cat_id, (0,0,0))
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
    print("====================================================================================")


if __name__ == "__main__":
    main()