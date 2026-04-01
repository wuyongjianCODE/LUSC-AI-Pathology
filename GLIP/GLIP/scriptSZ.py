import os
import numpy as np
import openslide
import cv2
import pandas as pd
import json
import random
from datetime import datetime
from openslide.deepzoom import DeepZoomGenerator
import argparse
import re
from PIL import Image


class COCODatasetGenerator:
    def __init__(self, images_dir, output_json_path):
        self.images_dir = images_dir
        self.output_json_path = output_json_path
        self.coco_format = self._init_coco_format()

    def _init_coco_format(self):
        return {
            "info": {
                "description": "Nuclei Dataset from Selected Patches",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "categories": [
                {"id": 1, "name": "Tumor Cell", "supercategory": "cell"},
                {"id": 2, "name": "Lymphocyte", "supercategory": "cell"},
                {"id": 3, "name": "Neutrophil", "supercategory": "cell"},
                {"id": 4, "name": "Eosinophil", "supercategory": "cell"},
                {"id": 5, "name": "Eosinophil Nucleus", "supercategory": "cell"},
                {"id": 6, "name": "Plasma Cell", "supercategory": "cell"},
                {"id": 7, "name": "Vascular Endothelial Cell", "supercategory": "cell"},
                {"id": 8, "name": "Stromal Cell", "supercategory": "cell"},
                {"id": 9, "name": "Histiocyte", "supercategory": "cell"}
            ],
            "images": [],
            "annotations": []
        }

    def _generate_random_circle(self, img_width, img_height):
        center_x = random.randint(int(img_width * 0.2), int(img_width * 0.8))
        center_y = random.randint(int(img_height * 0.2), int(img_height * 0.8))
        max_radius = min(img_width, img_height) // 10
        min_radius = min(img_width, img_height) // 20
        radius = random.randint(min_radius, max_radius)

        polygon = []
        for i in range(36):
            angle = i * 10
            rad = np.radians(angle)
            x = center_x + radius * np.cos(rad)
            y = center_y + radius * np.sin(rad)
            polygon.extend([float(x), float(y)])

        return {
            "polygon": polygon,
            "bbox": [center_x - radius, center_y - radius, radius * 2, radius * 2],
            "area": np.pi * (radius ** 2)
        }

    def generate_dataset(self):
        image_id, annotation_id = 1, 1
        for img_filename in os.listdir(self.images_dir):
            if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(self.images_dir, img_filename)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Skipping {img_filename}: {e}")
                continue

            self.coco_format["images"].append({
                "id": image_id, "width": width, "height": height,
                "file_name": img_filename, "license": 0, "date_captured": ""
            })

            circle_info = self._generate_random_circle(width, height)
            self.coco_format["annotations"].append({
                "id": annotation_id, "image_id": image_id, "category_id": 1,
                "segmentation": [circle_info["polygon"]], "area": circle_info["area"],
                "bbox": circle_info["bbox"], "iscrowd": 0
            })

            image_id += 1
            annotation_id += 1

        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_format, f, indent=4)
        print(f"COCO annotations saved to {self.output_json_path}")


def find_svs_in_subfolders(root_dir, candidate_names, thumb_filename):
    """
    优化：支持遍历软链接(ln -s)创建的文件夹，同时避免循环引用
    功能：深入所有层级的文件夹（包括软链接指向的目录）查找匹配的SVS文件
    """
    # 验证根目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误：WSI根目录不存在 → {root_dir}")
        return None

    all_found_svs = []  # 存储所有匹配的SVS文件
    visited_dirs = set()  # 记录已访问的真实目录（避免循环引用）

    # 自定义递归遍历函数（支持符号链接）
    def traverse(current_dir):
        # 解析当前目录的真实路径（处理符号链接）
        real_current = os.path.realpath(current_dir)

        # 检查是否已访问过该目录（避免循环）
        if real_current in visited_dirs:
            return
        visited_dirs.add(real_current)

        try:
            # 获取目录下所有条目
            entries = os.listdir(current_dir)
        except (PermissionError, OSError) as e:
            return

        for entry in entries:
            entry_path = os.path.join(current_dir, entry)

            # 检查是否为符号链接
            if os.path.islink(entry_path):
                # 解析符号链接指向的真实路径
                link_target = os.readlink(entry_path)
                real_target = os.path.abspath(os.path.join(current_dir, link_target))

                # 如果指向目录，则递归遍历
                if os.path.isdir(real_target):
                    traverse(real_target)
                continue  # 符号链接文件不处理

            # 如果是目录，递归遍历（非符号链接目录）
            if os.path.isdir(entry_path):
                traverse(entry_path)

            # 检查是否为SVS文件
            elif entry.lower().endswith(".svs"):
                svs_basename = os.path.splitext(entry)[0].lower()
                # 检查与候选名称的匹配度
                match_score = 0
                matched_candidate = None
                for candidate in candidate_names:
                    candidate_lower = candidate.lower()
                    # 优先级1：精确匹配
                    if candidate_lower == svs_basename:
                        match_score = 100
                        matched_candidate = candidate
                        break
                    # 优先级2：模糊包含
                    elif candidate_lower in svs_basename:
                        match_score = 50
                        matched_candidate = candidate
                        break
                    # 优先级3：前缀匹配
                    elif svs_basename.startswith(candidate_lower):
                        match_score = 30
                        matched_candidate = candidate
                        break
                if match_score > 0:
                    all_found_svs.append({
                        "path": entry_path,
                        "filename": entry,
                        "score": match_score,
                        "candidate": matched_candidate
                    })

    # 开始从根目录遍历
    traverse(root_dir)

    # 处理匹配结果
    if not all_found_svs:
        print("!!!!!!!!!!!!!!!!!!!!!!! 查找结果：未找到任何匹配的SVS文件 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return None
    else:
        # 按匹配得分排序，取最优结果
        all_found_svs.sort(key=lambda x: x["score"], reverse=True)
        best_match = all_found_svs[0]
        print(f"  匹配路径: {best_match['path']}")
        return best_match["path"]


def extract_wsi_name_candidates(thumb_filename):
    """
    提取WSI名称候选列表，针对医生命名特殊情况
    重点处理末尾包含2022的情况
    """
    # 去除文件扩展名
    base_name = os.path.splitext(thumb_filename)[0]

    # 提取_whole_classify之前的部分
    match = re.match(r'^(.*?)_whole_classify', base_name)
    if not match:
        prefix = base_name
    else:
        prefix = match.group(1)

    candidates = []

    # 模式1: 匹配类似1031703-8、1031703-9、1031703-10的格式
    pattern = r'^(\d+-(\d+))(?=\D|$)'
    id_match = re.match(pattern, prefix)
    if id_match:
        candidates.append(id_match.group(1))

    # 模式2: 处理包含多个连字符的情况，如1031703-9-92022...
    pattern2 = r'^(\d+-\d+)-'
    id_match2 = re.match(pattern2, prefix)
    if id_match2 and id_match2.group(1) not in candidates:
        candidates.append(id_match2.group(1))

    # 模式3: 原始前缀作为候选
    if prefix not in candidates:
        candidates.append(prefix)

    # 关键处理: 如果候选名称以2022结尾，添加去掉2022的版本作为新候选
    new_candidates = []
    for candidate in candidates:
        new_candidates.append(candidate)

        def get_prefix_before_third_dash(filename):
            # 分割字符串为部分列表
            parts = filename.split('-')

            # 检查是否有至少3个连字符
            if len(parts) >= 4:
                # 取前3部分并拼接（因为第三个连字符前有3个部分）
                return '-'.join(parts[:3])
            else:
                # 如果连字符不足3个，返回原始文件名（或根据需求处理）
                return filename

        candidate=get_prefix_before_third_dash(candidate)
        if (candidate.endswith('2022') or  candidate.endswith('2023') or candidate.endswith('2024')) and re.search(r'\d$', candidate[:-4]):
            trimmed = candidate  # 去掉末尾的2022
            if trimmed not in new_candidates:
                new_candidates.append(trimmed)
                print(f"Added trimmed candidate: {trimmed} (from {candidate})")

    return new_candidates[-1:]


def extract_thumbnail_roi(thumb_img, left_k=5, right_k=8, bottom_k=5):
    gray = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("No valid region detected")
        return None, (0, 0, 0, 0)

    max_contour = max(contours, key=cv2.contourArea)
    x_init, y_init, w_init, h_init = cv2.boundingRect(max_contour)

    x_shrink = x_init + left_k
    y_shrink = y_init
    w_shrink = w_init - left_k - right_k
    h_shrink = h_init - bottom_k

    x_shrink = max(x_shrink, 0)
    w_shrink = max(w_shrink, 10)
    h_shrink = max(h_shrink, 10)
    if x_shrink + w_shrink > thumb_img.shape[1]:
        w_shrink = thumb_img.shape[1] - x_shrink
    if y_shrink + h_shrink > thumb_img.shape[0]:
        h_shrink = thumb_img.shape[0] - y_shrink

    roi = thumb_img[y_shrink:y_shrink + h_shrink, x_shrink:x_shrink + w_shrink]
    return roi, (x_shrink, y_shrink, w_shrink, h_shrink)


def get_color_hsv_range(color_name):
    return {
        'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
        'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),
        'red2': (np.array([170, 50, 50]), np.array([180, 255, 255])),
        'white': (np.array([0, 0, 200]), np.array([180, 30, 255])),
        'green': (np.array([40, 50, 50]), np.array([80, 255, 255]))
    }[color_name]


def find_largest_connected_component(roi_img, color_name, min_area=1):
    lower, upper = get_color_hsv_range(color_name)
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None

    max_contour = max(valid_contours, key=cv2.contourArea)
    max_area = cv2.contourArea(max_contour)
    max_perimeter = cv2.arcLength(max_contour, closed=True)
    x, y, w, h = cv2.boundingRect(max_contour)

    component_mask = np.zeros_like(mask)
    cv2.drawContours(component_mask, [max_contour], -1, 255, -1)

    return {
        "bbox": (x, y, w, h), "center": (x + w // 2, y + h // 2),
        "area": max_area, "perimeter": max_perimeter,
        "mask": component_mask, "contour": max_contour
    }


def dilate_contour(contour, roi_img_shape, expand_pixel_roi):
    h, w = roi_img_shape[:2]
    orig_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(orig_mask, [contour], -1, 255, thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * expand_pixel_roi + 1, 2 * expand_pixel_roi + 1))
    dilated_mask = cv2.dilate(orig_mask, kernel, iterations=1)
    dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not dilated_contours:
        return None, dilated_mask
    return max(dilated_contours, key=cv2.contourArea), dilated_mask


def get_patch_range_from_contour(wsi, thumb_roi, blue_component, k, patch_size=224):
    roi_img, (roi_x_thumb, roi_y_thumb, _, _) = thumb_roi
    orig_contour = blue_component["contour"]
    wsi_w, wsi_h = wsi.level_dimensions[0]
    roi_h, roi_w = roi_img.shape[:2]

    scale_x, scale_y = wsi_w / roi_w, wsi_h / roi_h
    scale = (scale_x + scale_y) / 2

    # 计算肿瘤在WSI中的面积和周长
    tumor_area_wsi = blue_component["area"] * (scale ** 2)
    tumor_perimeter_wsi = blue_component["perimeter"] * scale

    expand_pixel_wsi = k * patch_size
    expand_pixel_roi = max(1, int(expand_pixel_wsi / scale))
    dilated_contour_roi, dilated_mask_roi = dilate_contour(orig_contour, roi_img.shape, expand_pixel_roi)

    if dilated_contour_roi is None:
        raise ValueError("Contour dilation failed")

    # 创建blue_component的原始mask（填实的外轮廓）并转换到WSI坐标
    blue_mask_roi = np.zeros_like(roi_img[:, :, 0])
    cv2.drawContours(blue_mask_roi, [orig_contour], -1, 255, -1)  # 填实外轮廓

    # 将blue_component的mask转换到WSI尺寸
    blue_mask_wsi = cv2.resize(blue_mask_roi, (wsi_w, wsi_h), interpolation=cv2.INTER_NEAREST)

    dilated_contour_wsi = np.array([[[int(p[0][0] * scale), int(p[0][1] * scale)]]
                                    for p in dilated_contour_roi], dtype=np.int32)
    x_wsi, y_wsi, w_wsi, h_wsi = cv2.boundingRect(dilated_contour_wsi)

    min_x_pixel = max(0, x_wsi - (x_wsi % patch_size))
    max_x_pixel = min(wsi_w, (x_wsi + w_wsi) + (patch_size - (x_wsi + w_wsi) % patch_size))
    min_y_pixel = max(0, y_wsi - (y_wsi % patch_size))
    max_y_pixel = min(wsi_h, (y_wsi + h_wsi) + (patch_size - (y_wsi + h_wsi) % patch_size))

    wsi_dilated_mask = np.zeros((wsi_h, wsi_w), dtype=np.uint8)
    cv2.drawContours(wsi_dilated_mask, [dilated_contour_wsi], -1, 255, thickness=cv2.FILLED)

    return (int(min_x_pixel // patch_size), int(max_x_pixel // patch_size),
            int(min_y_pixel // patch_size), int(max_y_pixel // patch_size),
            wsi_dilated_mask, scale, dilated_contour_roi, k, dilated_mask_roi,
            roi_x_thumb, roi_y_thumb, tumor_area_wsi, tumor_perimeter_wsi,
            blue_mask_wsi)  # 返回blue_component的WSI mask


def judge_thumbnail_color(thumb_img, thumb_x, thumb_y):
    thumb_h, thumb_w = thumb_img.shape[:2]
    thumb_x = max(0, min(thumb_x, thumb_w - 1))
    thumb_y = max(0, min(thumb_y, thumb_h - 1))

    half_patch = 1
    x_start, x_end = max(0, thumb_x - half_patch), min(thumb_w, thumb_x + half_patch)
    y_start, y_end = max(0, thumb_y - half_patch), min(thumb_h, thumb_y + half_patch)
    avg_color = cv2.mean(thumb_img[y_start:y_end, x_start:x_end])[:3]
    avg_color_bgr = np.array(avg_color, dtype=np.uint8).reshape(1, 1, 3)

    hsv = cv2.cvtColor(avg_color_bgr, cv2.COLOR_BGR2HSV)
    lower_white, upper_white = get_color_hsv_range('white')
    lower_blue, upper_blue = get_color_hsv_range('blue')

    if cv2.inRange(hsv, lower_white, upper_white).any() or cv2.inRange(hsv, lower_blue, upper_blue).any():
        return None

    lower_red1, upper_red1 = get_color_hsv_range('red')
    lower_red2, upper_red2 = get_color_hsv_range('red2')
    lower_green, upper_green = get_color_hsv_range('green')

    is_red = cv2.inRange(hsv, lower_red1, upper_red1).any() or cv2.inRange(hsv, lower_red2, upper_red2).any()
    is_green = cv2.inRange(hsv, lower_green, upper_green).any()

    return 'R' if is_red else 'G' if is_green else 'R'


def get_closest_contour_point(contour, point):
    min_dist = float('inf')
    closest_point = None
    for p in contour:
        px, py = p[0][0], p[0][1]
        dist = np.hypot(px - point[0], py - point[1])
        if dist < min_dist:
            min_dist, closest_point = dist, (px, py)
    return (closest_point[0] - point[0], closest_point[1] - point[1]) if closest_point else (0, 0)


def get_patch_range_from_center(wsi, center_roi, thumb_roi, k, patch_size=224):
    roi_img, (roi_x_thumb, roi_y_thumb, _, _) = thumb_roi
    wsi_w, wsi_h = wsi.level_dimensions[0]
    roi_w, roi_h = roi_img.shape[1], roi_img.shape[0]

    scale_x, scale_y = wsi_w / roi_w, wsi_h / roi_h
    scale = (scale_x + scale_y) / 2

    center_x_roi, center_y_roi = center_roi
    center_x_wsi = (center_x_roi) * scale
    center_y_wsi = (center_y_roi) * scale

    expand_pixel = k * patch_size
    min_x_pixel = max(0, center_x_wsi - expand_pixel)
    max_x_pixel = min(wsi_w, center_x_wsi + expand_pixel)
    min_y_pixel = max(0, center_y_wsi - expand_pixel)
    max_y_pixel = min(wsi_h, center_y_wsi + expand_pixel)

    return (int(min_x_pixel // patch_size), int(max_x_pixel // patch_size),
            int(min_y_pixel // patch_size), int(max_y_pixel // patch_size),
            scale, roi_x_thumb, roi_y_thumb)


def process_dataset(pred_dir, wsi_dir, args, results, patch_save_dir):
    for thumb_filename in os.listdir(pred_dir):
        # 检查是否为分类缩略图
        if not (thumb_filename.lower().endswith(('.png', '.jpg', '.jpeg')) and
                '_whole_classify' in thumb_filename):
            continue

        # 提取WSI名称候选列表（包含去除2022的版本）
        wsi_candidates = extract_wsi_name_candidates(thumb_filename)
        print(f"Generated candidates for {thumb_filename}: {wsi_candidates}")

        # 查找对应的WSI文件，尝试所有候选名称
        svs_path = find_svs_in_subfolders(wsi_dir, wsi_candidates, thumb_filename)
        if not svs_path or not os.path.exists(svs_path):
            print(f"WSI not found for {thumb_filename} with candidates {wsi_candidates}, skipping")
            continue

        print(f"Processing {thumb_filename} -> {os.path.basename(svs_path)}")
        thumb_img = cv2.imread(os.path.join(pred_dir, thumb_filename))
        if thumb_img is None:
            print(f"Cannot read thumbnail {thumb_filename}")
            continue

        thumb_roi = extract_thumbnail_roi(thumb_img)
        if not thumb_roi:
            print(f"No valid ROI in {thumb_filename}")
            continue
        roi_img, (roi_x_thumb, roi_y_thumb, roi_w_thumb, roi_h_thumb) = thumb_roi
        roi_w, roi_h = roi_img.shape[1], roi_img.shape[0]

        blue_component = find_largest_connected_component(roi_img, 'blue', min_area=1)
        region_type, patch_range, reference_point = None, None, None
        wsi_dilated_mask, scale, current_k = None, 1.0, args.k
        dilated_mask_roi, dilated_contour_roi = None, None
        blue_mask_wsi = None  # 存储blue_component的WSI mask
        selected_patches = []
        roi_x_thumb_global, roi_y_thumb_global = roi_x_thumb, roi_y_thumb
        tumor_area_wsi, tumor_perimeter_wsi = 0, 0

        try:
            if blue_component:
                region_type = "Non-blue area around tumor"
                slide = openslide.open_slide(svs_path)
                patch_range = get_patch_range_from_contour(
                    slide, thumb_roi, blue_component, k=args.k)
                # 解析返回值，包含blue_mask_wsi
                (min_x_patch, max_x_patch, min_y_patch, max_y_patch, wsi_dilated_mask,
                 scale, dilated_contour_roi, current_k, dilated_mask_roi, roi_x_thumb_global,
                 roi_y_thumb_global, tumor_area_wsi, tumor_perimeter_wsi,
                 blue_mask_wsi) = patch_range
                reference_point = blue_component['center']
            else:
                red_components = []
                red_component1 = find_largest_connected_component(roi_img, 'red')
                red_component2 = find_largest_connected_component(roi_img, 'red2')
                if red_component1: red_components.append(red_component1)
                if red_component2: red_components.append(red_component2)

                if red_components:
                    red_component = max(red_components, key=lambda x: x['area'])
                    region_type = "Area around tumor bed"
                    slide = openslide.open_slide(svs_path)
                    patch_range = get_patch_range_from_center(
                        slide, red_component['center'], thumb_roi, k=args.k)
                    min_x_patch, max_x_patch, min_y_patch, max_y_patch, scale, roi_x_thumb_global, roi_y_thumb_global = patch_range
                    reference_point = red_component['center']
                else:
                    region_type = "Area around WSI center"
                    center_roi = (roi_w // 2, roi_h // 2)
                    slide = openslide.open_slide(svs_path)
                    patch_range = get_patch_range_from_center(
                        slide, center_roi, thumb_roi, k=args.k)
                    min_x_patch, max_x_patch, min_y_patch, max_y_patch, scale, roi_x_thumb_global, roi_y_thumb_global = patch_range
                    reference_point = center_roi

            if 'slide' not in locals():
                slide = openslide.open_slide(svs_path)
            wsi_w, wsi_h = slide.level_dimensions[0]
            data_gen = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)
            level = data_gen.level_count - 1

            if not patch_range:
                print("No patch range calculated")
                continue

            ref_x_roi, ref_y_roi = reference_point
            ref_x_wsi = (roi_x_thumb_global + ref_x_roi) * scale
            ref_y_wsi = (roi_y_thumb_global + ref_y_roi) * scale

            patch_coords = []
            for x in range(min_x_patch, max_x_patch):
                for y in range(min_y_patch, max_y_patch):
                    patch_center_x_wsi = x * 224 + 112
                    patch_center_y_wsi = y * 224 + 112
                    distance = abs(patch_center_x_wsi - ref_x_wsi) + abs(patch_center_y_wsi - ref_y_wsi)
                    patch_coords.append((distance, x, y))

            patch_coords.sort()
            total_checked, saved_patches = 0, 0

            for _, x_patch, y_patch in patch_coords:
                total_checked += 1
                if len(selected_patches) >= args.thre:
                    break

                try:
                    in_dilated_area = True
                    if blue_component and wsi_dilated_mask is not None:
                        patch_min_x = x_patch * 224
                        patch_max_x = (x_patch + 1) * 224
                        patch_min_y = y_patch * 224
                        patch_max_y = (y_patch + 1) * 224

                        # 检查patch是否在blue_component的mask内部
                        if blue_mask_wsi is not None:
                            # 获取patch在WSI中的区域
                            mask_roi_min_x = max(0, patch_min_x)
                            mask_roi_max_x = min(blue_mask_wsi.shape[1], patch_max_x)
                            mask_roi_min_y = max(0, patch_min_y)
                            mask_roi_max_y = min(blue_mask_wsi.shape[0], patch_max_y)

                            # 检查patch区域是否与blue_component有重叠
                            if mask_roi_max_x > mask_roi_min_x and mask_roi_max_y > mask_roi_min_y:
                                blue_overlap = blue_mask_wsi[mask_roi_min_y:mask_roi_max_y,
                                               mask_roi_min_x:mask_roi_max_x]
                                # 如果有任何像素属于blue_component，则排除该patch
                                if np.sum(blue_overlap) > 0:
                                    continue  # 跳过位于blue_component内部的patch

                        # 检查patch是否在膨胀区域内
                        mask_roi_min_x = max(0, patch_min_x)
                        mask_roi_max_x = min(wsi_dilated_mask.shape[1], patch_max_x)
                        mask_roi_min_y = max(0, patch_min_y)
                        mask_roi_max_y = min(wsi_dilated_mask.shape[0], patch_max_y)

                        if mask_roi_max_x <= mask_roi_min_x or mask_roi_max_y <= mask_roi_min_y:
                            in_dilated_area = False
                        else:
                            overlap_mask = wsi_dilated_mask[mask_roi_min_y:mask_roi_max_y,
                                           mask_roi_min_x:mask_roi_max_x]
                            if np.sum(overlap_mask) == 0:
                                in_dilated_area = False

                    if not in_dilated_area:
                        continue

                    patch_center_x_wsi = x_patch * 224 + 112
                    patch_center_y_wsi = y_patch * 224 + 112
                    patch_center_x_roi = patch_center_x_wsi / scale
                    patch_center_y_roi = patch_center_y_wsi / scale
                    patch_center_x_thumb = roi_x_thumb_global + patch_center_x_roi
                    patch_center_y_thumb = roi_y_thumb_global + patch_center_y_roi

                    patch_color_code = judge_thumbnail_color(
                        thumb_img, int(round(patch_center_x_thumb)), int(round(patch_center_y_thumb)))
                    if not patch_color_code:
                        continue

                    vx, vy = 0, 0
                    if blue_component:
                        patch_center_roi = (patch_center_x_roi, patch_center_y_roi)
                        vx_roi, vy_roi = get_closest_contour_point(blue_component["contour"], patch_center_roi)
                        vx, vy = int(round(vx_roi * scale)), int(round(vy_roi * scale))

                    patch_rgb = np.array(data_gen.get_tile(level, (x_patch, y_patch)))
                    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
                    distance = np.sqrt((patch_center_x_wsi - ref_x_wsi) ** 2 +
                                       (patch_center_y_wsi - ref_y_wsi) ** 2)

                    selected_patches.append({
                        "x": x_patch, "y": y_patch, "patch_rgb": patch_rgb, "patch_bgr": patch_bgr,
                        "distance": distance, "color_code": patch_color_code,
                        "thumb_x": int(round(patch_center_x_thumb)),
                        "thumb_y": int(round(patch_center_y_thumb)),
                        "vector_to_tumor": (vx, vy),
                        "tumor_area": tumor_area_wsi,
                        "tumor_perimeter": tumor_perimeter_wsi
                    })

                except Exception as e:
                    print(f"Patch (x={x_patch}, y={y_patch}) error: {e}")
                    continue

            # 提取用于生成patch文件名的基础WSI名称
            base_wsi_name = os.path.splitext(os.path.basename(svs_path))[0]

            for patch in selected_patches:
                vx, vy = patch["vector_to_tumor"]
                area = int(round(patch["tumor_area"]))
                perimeter = int(round(patch["tumor_perimeter"]))
                patch_filename = (f"{base_wsi_name}_patch_x{patch['x']}_y{patch['y']}_color{patch['color_code']}_"
                                  f"TargetTumorArea{area}_Perimeter{perimeter}_"
                                  f"vector2targetbyVX{vx}VY{vy}.png")
                patch_save_path = os.path.join(patch_save_dir, patch_filename)
                cv2.imwrite(patch_save_path, patch['patch_bgr'])

                results.append({
                    "wsi_name": os.path.basename(svs_path), "patch_x": patch['x'], "patch_y": patch['y'],
                    "patch_filename": patch_filename, "patch_color": patch['color_code'],
                    "distance_to_reference": patch['distance'],
                    "vector_to_tumor_vx": vx, "vector_to_tumor_vy": vy,
                    "tumor_area": area, "tumor_perimeter": perimeter,
                    "region_type": region_type
                })
                saved_patches += 1

            print(f"Checked {total_checked} patches, saved {saved_patches}/{args.thre}")
            slide.close()

        except Exception as e:
            print(f"Error processing {svs_path}: {e}")
            if 'slide' in locals():
                slide.close()
            continue


def main(args):
    # 只处理深圳数据集
    # dataset_pairs = [
    #     (args.pred_dir, args.wsi_dir)
    # ]
    #
    # # 创建保存目录
    patch_save_dir = "selected_patches"
    # if os.path.exists(patch_save_dir):
    #     os.system(f'rm -r {patch_save_dir}')
    # os.makedirs(patch_save_dir, exist_ok=True)
    # results = []
    #
    # # 处理数据集
    # for pred_dir, wsi_dir in dataset_pairs:
    #     if os.path.exists(pred_dir) and os.path.exists(wsi_dir):
    #         print(f"\nProcessing dataset: {pred_dir} -> {wsi_dir}")
    #         process_dataset(pred_dir, wsi_dir, args, results, patch_save_dir)
    #     else:
    #         print(f"Skipping invalid dataset: {pred_dir} or {wsi_dir} does not exist")
    #
    # # 生成COCO标注
    # print("\nGenerating combined COCO annotations...")
    # coco_generator = COCODatasetGenerator(patch_save_dir,
    #                                       os.path.join(patch_save_dir, "coco_annotations.json"))
    # coco_generator.generate_dataset()
    #
    # # 保存结果并传输文件
    # if results:
    #     pd.DataFrame(results).to_excel("patch_details.xlsx", index=False)
    # print(f"Total patches processed: {len(results)}")

    # 传输到目标服务器：先删除远程文件夹，再上传新的
    remote_server = "root@192.168.103.72"
    remote_dir = "/home/data/jy/GLIP/DATASET/selected_patches"

    # 删除远程服务器上的文件夹
    print(f"Removing remote directory: {remote_dir}")
    os.system(f'ssh {remote_server} "rm -rf {remote_dir}"')

    # 上传新的文件夹
    print(f"Uploading to {remote_server}:{remote_dir}")
    os.system(f'scp -r {patch_save_dir} {remote_server}:{os.path.dirname(remote_dir)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WSI patch extraction tool')

    # 深圳数据集目录参数
    parser.add_argument('--pred_dir', type=str,
                        default='/data4/TOSHOW_MULTISCALE_shenzhen/',
                        help='Thumbnail directory (Shenzhen dataset)')
    parser.add_argument('--wsi_dir', type=str,
                        default='/data4/深圳分院35例/',
                        help='WSI directory (Shenzhen dataset)')

    # 其他参数
    parser.add_argument('--k', type=int, default=5, help='Expanded patch width')
    parser.add_argument('--thre', type=int, default=1000000, help='Max patches per WSI')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    main(args)
