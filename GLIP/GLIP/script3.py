import os
import numpy as np
import openslide
import matplotlib.pyplot as plt
import cv2
import torch
import pandas as pd
import json
import random
from datetime import datetime
from openslide.deepzoom import DeepZoomGenerator
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from PIL import Image
import torchvision.transforms as transforms

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


class COCODatasetGenerator:
    """COCO数据集生成器，用于将patch文件夹转换为COCO格式标注"""

    def __init__(self, images_dir, output_json_path):
        self.images_dir = images_dir
        self.output_json_path = output_json_path
        self.coco_format = self._init_coco_format()

    def _init_coco_format(self):
        """初始化COCO格式的基本结构"""
        return {
            "info": {
                "description": "Nuclei Dataset from Selected Patches",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "categories": [
                {
                    "id": 1,
                    "name": "Stroma",
                    "supercategory": "nuclei"
                },
                {
                    "id": 2,
                    "name": "Tumor",
                    "supercategory": "nuclei"
                }

            ],
            "images": [],
            "annotations": []
        }

    def _generate_random_circle(self, img_width, img_height):
        """在图片中随机生成一个圆形标注（转换为多边形表示）"""
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
            polygon.append(float(x))
            polygon.append(float(y))

        return {
            "polygon": polygon,
            "bbox": [
                center_x - radius,
                center_y - radius,
                radius * 2,
                radius * 2
            ],
            "area": np.pi * (radius ** 2)
        }

    def generate_dataset(self):
        """生成COCO格式数据集"""
        image_id = 1
        annotation_id = 1

        for img_filename in os.listdir(self.images_dir):
            if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(self.images_dir, img_filename)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"跳过无效图片 {img_filename}: {e}")
                continue

            self.coco_format["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": img_filename,
                "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            circle_info = self._generate_random_circle(width, height)
            self.coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [circle_info["polygon"]],
                "area": circle_info["area"],
                "bbox": circle_info["bbox"],
                "iscrowd": 0
            })

            print(f"已处理标注: {img_filename} (图像ID: {image_id})")
            image_id += 1
            annotation_id += 1

        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_format, f, indent=4)

        print(f"\nCOCO数据集生成完成！共处理 {image_id - 1} 张图片")
        print(f"标注文件保存路径: {self.output_json_path}")


def find_svs_in_subfolders(root_dir, target_name):
    """在根目录及其所有子文件夹中查找指定名称的SVS文件"""
    target_svs = f"{target_name}.svs"
    # 递归遍历所有子文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == target_svs:
                return os.path.join(dirpath, filename)

    # 模糊匹配（包含目标名称）
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.svs') and target_name in filename:
                print(f"找到模糊匹配的SVS文件: {filename} (目标: {target_svs})")
                return os.path.join(dirpath, filename)

    return None


def extract_thumbnail_roi(thumb_img):
    """提取缩略图中有效图像区域（去除坐标轴和空白）"""
    # 转换为灰度图，检测有效区域边界
    gray = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 非背景区域（<240）为有效区域
    # 查找有效区域连通域
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # 无有效区域

    # 选择最大连通域作为有效区域
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # 裁剪有效区域
    roi = thumb_img[y:y + h, x:x + w]
    return roi, (x, y, w, h)  # 返回裁剪区域和原始坐标


def get_color_hsv_range(color_name):
    """获取指定颜色的HSV范围（用于连通域检测）"""
    color_ranges = {
        'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),  # 蓝色（肿瘤）
        'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),  # 红色范围1（瘤床）
        'red2': (np.array([170, 50, 50]), np.array([180, 255, 255])),  # 红色范围2（瘤床）
        'white': (np.array([0, 0, 200]), np.array([180, 30, 255])),  # 白色（背景）
        'green': (np.array([40, 50, 50]), np.array([80, 255, 255]))  # 绿色（坏死）
    }
    return color_ranges[color_name]


def find_largest_connected_component(roi_img, color_name, min_area=50):
    """
    使用连通域算法找到指定颜色的最大连通域
    :param roi_img: 有效区域图像
    :param color_name: 目标颜色名称（'blue'/'red'等）
    :param min_area: 最小连通域面积（过滤噪声）
    :return: 最大连通域信息（边界框、中心点、面积、掩码）
    """
    # 1. 颜色阈值分割获取二值掩码
    lower, upper = get_color_hsv_range(color_name)
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)  # 目标颜色区域为255，其他为0

    # 2. 查找连通域（核心连通域函数）
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,  # 只检测外轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓
    )

    if not contours:
        return None  # 无目标颜色连通域

    # 3. 过滤小面积连通域（去噪）并找到最大连通域
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None  # 所有连通域均为噪声

    max_contour = max(valid_contours, key=cv2.contourArea)
    max_area = cv2.contourArea(max_contour)

    # 4. 计算连通域边界框和中心点
    x, y, w, h = cv2.boundingRect(max_contour)
    center_x = x + w // 2
    center_y = y + h // 2

    # 5. 生成最大连通域的掩码（仅保留最大连通域）
    component_mask = np.zeros_like(mask)
    cv2.drawContours(component_mask, [max_contour], -1, 255, -1)  # 填充最大连通域

    return {
        "bbox": (x, y, w, h),  # 连通域边界框（x,y,w,h）
        "center": (center_x, center_y),  # 连通域中心点坐标
        "area": max_area,  # 连通域面积
        "mask": component_mask  # 连通域掩码（仅最大连通域为255）
    }


def is_blue_patch(patch_img, blue_threshold=0.3):
    """判断patch是否为蓝色连通域（肿瘤），蓝色占比超过阈值则视为蓝色patch"""
    lower, upper = get_color_hsv_range('blue')
    hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, lower, upper)

    # 计算蓝色区域占比（连通域面积/总像素）
    blue_ratio = np.sum(blue_mask > 0) / (blue_mask.shape[0] * blue_mask.shape[1])
    return blue_ratio > blue_threshold


def get_patch_range_from_bbox(wsi, thumb_roi, bbox, k, patch_size=224):
    """从连通域边界框计算外围k个patch的范围"""
    # 原始WSI尺寸
    wsi_w, wsi_h = wsi.level_dimensions[0]
    # 缩略图有效区域尺寸
    roi_w = thumb_roi[1][2]
    roi_h = thumb_roi[1][3]

    # 计算缩放比例（原始WSI像素 / 缩略图像素）
    scale_x = wsi_w / roi_w
    scale_y = wsi_h / roi_h
    if abs(scale_x - scale_y) > 0.1:
        print(f"警告：缩略图比例与WSI不一致，可能导致坐标偏差 (x缩放: {scale_x:.2f}, y缩放: {scale_y:.2f})")
    scale = (scale_x + scale_y) / 2  # 取平均比例

    # 连通域在原始WSI中的像素坐标
    x_roi, y_roi, w_roi, h_roi = bbox
    x_wsi = x_roi * scale
    y_wsi = y_roi * scale
    w_wsi = w_roi * scale
    h_wsi = h_roi * scale

    # 扩展k个patch宽度（每个patch 224像素）
    expand_pixel = k * patch_size
    min_x_pixel = max(0, x_wsi - expand_pixel)
    max_x_pixel = min(wsi_w, x_wsi + w_wsi + expand_pixel)
    min_y_pixel = max(0, y_wsi - expand_pixel)
    max_y_pixel = min(wsi_h, y_wsi + h_wsi + expand_pixel)

    # 转换为patch索引（x_patch = 像素x // 224）
    return (
        int(min_x_pixel // patch_size),
        int(max_x_pixel // patch_size),
        int(min_y_pixel // patch_size),
        int(max_y_pixel // patch_size)
    )


def get_patch_range_from_center(wsi, center_roi, thumb_roi, k, patch_size=224):
    """从连通域中心点计算周围k个patch的范围"""
    # 原始WSI尺寸
    wsi_w, wsi_h = wsi.level_dimensions[0]
    # 缩略图有效区域尺寸
    roi_w = thumb_roi[1][2]
    roi_h = thumb_roi[1][3]

    # 计算缩放比例
    scale_x = wsi_w / roi_w
    scale_y = wsi_h / roi_h
    scale = (scale_x + scale_y) / 2  # 取平均比例

    # 中心点在原始WSI中的像素坐标
    center_x_roi, center_y_roi = center_roi
    center_x_wsi = center_x_roi * scale
    center_y_wsi = center_y_roi * scale

    # 扩展k个patch宽度
    expand_pixel = k * patch_size
    min_x_pixel = max(0, center_x_wsi - expand_pixel)
    max_x_pixel = min(wsi_w, center_x_wsi + expand_pixel)
    min_y_pixel = max(0, center_y_wsi - expand_pixel)
    max_y_pixel = min(wsi_h, center_y_wsi + expand_pixel)

    # 转换为patch索引
    return (
        int(min_x_pixel // patch_size),
        int(max_x_pixel // patch_size),
        int(min_y_pixel // patch_size),
        int(max_y_pixel // patch_size)
    )


def load_glip_model(config_file, weight_path):
    """加载GLIP模型（如需细胞检测）"""
    cfg.merge_from_file(config_file)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(f"加载GLIP模型: {weight_path}")

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    if weight_path:
        _ = checkpointer.load(weight_path, force=True)

    model.eval()
    return model, cfg


def detect_cells(model, image, cfg, prompt, confidence_threshold=0.5):
    """使用GLIP模型检测细胞数量"""
    input_tensor = preprocess_image(image, cfg)

    with torch.no_grad():
        outputs = model(input_tensor, captions=['Stroma. Tumor.'])

    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()

    cell_indices = (scores > confidence_threshold)
    return sum(cell_indices), boxes[cell_indices], scores[cell_indices]


def preprocess_image(image, cfg):
    """预处理图像用于GLIP检测"""
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize((min_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN,
            std=cfg.INPUT.PIXEL_STD
        )
    ])
    return transform(image).unsqueeze(0).to(cfg.MODEL.DEVICE)


def main(args):
    if args.command == "detect":
        # 创建保存目录
        patch_save_dir = "selected_patches"
        mkdir(patch_save_dir)

        # 缩略图目录和WSI目录（默认值已配置）
        pred_dir = args.pred_dir
        SVSDIR = args.wsi_dir
        print(f"开始处理缩略图，目录：{pred_dir}")
        print(f"在WSI目录及其子文件夹中查找：{SVSDIR}")
        print(f"每张WSI保留的最大patch数量：{args.thre}")

        # 结果存储列表
        results = []

        # 遍历所有缩略图
        for thumb_filename in os.listdir(pred_dir):
            if not thumb_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 提取WSI名称（根据缩略图命名规则）
            if '_whole_classify_mpr' in thumb_filename:
                wsi_name = thumb_filename.split('_whole_classify_mpr')[0]
            else:
                print(f"无法解析缩略图文件名：{thumb_filename}，跳过")
                continue

            # 查找对应的WSI文件（包括子文件夹）
            svs_path = find_svs_in_subfolders(SVSDIR, wsi_name)
            if not svs_path or not os.path.exists(svs_path):
                print(f"未找到对应的WSI文件（{wsi_name}.svs），跳过缩略图{thumb_filename}")
                continue

            print(f"\n处理缩略图：{thumb_filename}，对应WSI：{os.path.basename(svs_path)}")
            print(f"WSI路径：{svs_path}")

            # 读取缩略图并提取有效区域
            thumb_img = cv2.imread(os.path.join(pred_dir, thumb_filename))
            if thumb_img is None:
                print(f"无法读取缩略图：{thumb_filename}")
                continue

            thumb_roi = extract_thumbnail_roi(thumb_img)
            if not thumb_roi:
                print(f"缩略图{thumb_filename}无有效图像区域，跳过")
                continue
            roi_img, roi_coords = thumb_roi  # roi_coords: (x, y, w, h)
            roi_w, roi_h = roi_img.shape[1], roi_img.shape[0]

            # 1. 查找最大蓝色连通域（肿瘤）
            blue_component = find_largest_connected_component(roi_img, 'blue', min_area=100)
            region_type = None
            patch_range = None
            reference_point = None  # 用于计算patch距离的参考点

            if blue_component:
                region_type = "蓝色连通域（肿瘤）外围非蓝色区域"
                print(f"找到最大{region_type}，面积：{blue_component['area']}，边界框：{blue_component['bbox']}")
                # 计算蓝色连通域外围k个patch的范围
                patch_range = get_patch_range_from_bbox(
                    openslide.open_slide(svs_path),
                    thumb_roi,
                    blue_component['bbox'],
                    k=args.k,
                    patch_size=224
                )
                # 以蓝色连通域中心作为参考点
                reference_point = blue_component['center']
            else:
                # 2. 无蓝色连通域，查找最大红色连通域（瘤床）
                red_component1 = find_largest_connected_component(roi_img, 'red', min_area=100)
                red_component2 = find_largest_connected_component(roi_img, 'red2', min_area=100)

                # 合并两个红色范围的连通域并选最大
                red_components = []
                if red_component1:
                    red_components.append(red_component1)
                if red_component2:
                    red_components.append(red_component2)

                if red_components:
                    red_component = max(red_components, key=lambda x: x['area'])
                    region_type = "红色连通域（瘤床）周围区域"
                    print(f"未找到蓝色连通域，找到最大{region_type}，面积：{red_component['area']}，中心点：{red_component['center']}")
                    # 计算红色连通域中心周围k个patch的范围
                    patch_range = get_patch_range_from_center(
                        openslide.open_slide(svs_path),
                        red_component['center'],
                        thumb_roi,
                        k=args.k,
                        patch_size=224
                    )
                    reference_point = red_component['center']
                else:
                    # 3. 无蓝无红连通域，使用WSI中心点
                    region_type = "WSI中心周围区域"
                    center_roi = (roi_w // 2, roi_h // 2)
                    print(f"未找到蓝色和红色连通域，使用{region_type}，缩略图中心点：{center_roi}")
                    # 计算WSI中心周围k个patch的范围
                    patch_range = get_patch_range_from_center(
                        openslide.open_slide(svs_path),
                        center_roi,
                        thumb_roi,
                        k=args.k,
                        patch_size=224
                    )
                    reference_point = center_roi

            # 提取并保存符合条件的patch（带数量上限和早期终止）
            try:
                slide = openslide.open_slide(svs_path)
                wsi_w, wsi_h = slide.level_dimensions[0]
                print(f"WSI尺寸：宽度={wsi_w}，高度={wsi_h}")

                # 创建DeepZoom生成器（用于提取patch）
                data_gen = DeepZoomGenerator(
                    slide,
                    tile_size=224,
                    overlap=0,
                    limit_bounds=False
                )
                level = data_gen.level_count - 1  # 最高分辨率层级

                if not patch_range:
                    print("无法计算patch范围，跳过")
                    continue

                min_x_patch, max_x_patch, min_y_patch, max_y_patch = patch_range
                print(f"基于{region_type}，需处理的patch范围：x[{min_x_patch}:{max_x_patch}], y[{min_y_patch}:{max_y_patch}]")
                print(f"找到{args.thre}个符合条件的patch后将停止处理")

                # 加载GLIP模型（如需细胞检测）
                model, cfg = None, None
                if args.enable_detection:
                    print("加载GLIP模型...")
                    model, cfg = load_glip_model(args.config_file, args.weight_path)
                    print("GLIP模型加载完成")

                # 收集符合条件的patch（非蓝色）
                selected_patches = []
                total_checked = 0

                # 计算参考点在WSI中的像素坐标（用于距离排序）
                ref_x_wsi = reference_point[0] * (wsi_w / roi_w)
                ref_y_wsi = reference_point[1] * (wsi_h / roi_h)

                # 按距离参考点由近及远的顺序遍历patch（优化搜索顺序）
                # 计算参考点对应的patch坐标
                ref_x_patch = int(ref_x_wsi // 224)
                ref_y_patch = int(ref_y_wsi // 224)

                # 生成按距离排序的patch坐标列表（螺旋式向外搜索）
                patch_coords = []
                for dx in range(-args.k, args.k + 1):
                    for dy in range(-args.k, args.k + 1):
                        x = ref_x_patch + dx
                        y = ref_y_patch + dy
                        # 检查是否在有效范围内
                        if (min_x_patch <= x <= max_x_patch and
                                min_y_patch <= y <= max_y_patch):
                            # 计算距离
                            distance = abs(dx) + abs(dy)  # 曼哈顿距离，快速计算
                            patch_coords.append((distance, x, y))

                # 按距离排序，优先检查离参考点近的patch
                patch_coords.sort()

                # 遍历排序后的patch，找到足够数量后立即停止
                for _, x_patch, y_patch in patch_coords:
                    total_checked += 1

                    # 达到数量上限，提前终止
                    if len(selected_patches) >= args.thre:
                        break

                    try:
                        # 提取patch
                        patch_rgb = np.array(data_gen.get_tile(level, (x_patch, y_patch)))
                        patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)  # 转换为BGR用于颜色判断

                        # 判断是否为非蓝色patch
                        if blue_component is None or not is_blue_patch(patch_bgr, args.blue_threshold):
                            # 计算精确距离
                            patch_center_x = x_patch * 224 + 112
                            patch_center_y = y_patch * 224 + 112
                            distance = np.sqrt(
                                (patch_center_x - ref_x_wsi) ** 2 +
                                (patch_center_y - ref_y_wsi) ** 2
                            )

                            selected_patches.append({
                                "x": x_patch,
                                "y": y_patch,
                                "patch_rgb": patch_rgb,
                                "patch_bgr": patch_bgr,
                                "distance": distance
                            })

                    except Exception as e:
                        print(f"提取patch (x={x_patch}, y={y_patch}) 失败：{e}")
                        continue

                # 保存选中的patch
                saved_patches = len(selected_patches)
                for patch in selected_patches:
                    patch_filename = f"{wsi_name}_patch_x{patch['x']}_y{patch['y']}.png"
                    patch_save_path = os.path.join(patch_save_dir, patch_filename)
                    cv2.imwrite(patch_save_path, patch['patch_bgr'])

                    # 细胞检测（如启用）
                    cell_count = 0
                    if args.enable_detection and model is not None:
                        cell_count, _, _ = detect_cells(
                            model, patch['patch_rgb'], cfg, args.prompt, args.confidence_threshold
                        )

                    results.append({
                        "wsi_name": os.path.basename(svs_path),
                        "patch_x": patch['x'],
                        "patch_y": patch['y'],
                        "patch_filename": patch_filename,
                        "distance_to_reference": patch['distance'],
                        "cell_count": cell_count,
                        "region_type": region_type
                    })

                print(f"处理完成：检查了{total_checked}个patch，保存了{saved_patches}个（上限值：{args.thre}）")

            except Exception as e:
                print(f"处理WSI {svs_path} 时出错：{e}")
                continue

        # 保存检测结果到Excel
        if args.enable_detection and results:
            df = pd.DataFrame(results)
            wsi_summary = df.groupby(["wsi_name", "region_type"])["cell_count"].sum().reset_index()
            wsi_summary.rename(columns={"cell_count": "total_cell_count"}, inplace=True)

            df.to_excel("cell_detection_details.xlsx", index=False)
            wsi_summary.to_excel("cell_detection_summary.xlsx", index=False)
            print("检测结果已保存到Excel")
            print("汇总统计:\n", wsi_summary)
        else:
            if not results:
                print("未处理任何有效patch")

        # 生成COCO数据集
        print("\n生成COCO格式标注...")
        coco_generator = COCODatasetGenerator(patch_save_dir, os.path.join(patch_save_dir, "coco_annotations.json"))
        coco_generator.generate_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于连通域的WSI patch提取工具（带早期终止机制）')

    # 核心参数（默认值已配置）
    parser.add_argument('--pred_dir', type=str,
                        default='/data4/TOSHOW_MULTISCALE_shenzhen/',
                        help='缩略图目录（默认：/data4/TOSHOW_MULTISCALE_shenzhen/）')
    parser.add_argument('--wsi_dir', type=str,
                        default='/data4/深圳分院35例/',
                        help='WSI根目录（默认：/data4/深圳分院35例/）')
    parser.add_argument('--k', type=int, default=4,
                        help='外围扩展的patch宽度（默认：4）')
    parser.add_argument('--thre', type=int, default=1,
                        help='每张WSI保留的最大patch数量（默认：1）')
    parser.add_argument('--blue-threshold', type=float, default=0.3,
                        help='蓝色patch判断阈值（占比>该值视为蓝色，默认：0.3）')

    # 其他参数
    parser.add_argument('--command', type=str, default='detect',
                        help='命令类型（默认：detect）')
    parser.add_argument('--start', type=str, default='0',
                        help='开始处理的ID（默认：0）')
    parser.add_argument('--show', action='store_false',
                        help='是否显示结果（默认：不显示）')
    parser.add_argument('--enable_detection', action='store_false',
                        help='是否启用细胞检测（默认：不启用）')
    parser.add_argument("--config-file", type=str, default='./config.yaml',
                        help="GLIP配置文件路径（默认：./config.yaml）")
    parser.add_argument("--weight-path", type=str, default='./glip_model.pth',
                        help="GLIP权重路径（默认：./glip_model.pth）")
    parser.add_argument("--prompt", default='cell.',
                        help="GLIP提示文本（默认：cell.）")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="检测置信度阈值（默认：0.5）")

    args = parser.parse_args()
    os.system('rm -r  /data4/wyj/MRCNN/selected_patches')
    main(args)
    os.system('scp -r /data4/wyj/MRCNN/selected_patches root@192.168.103.72:/home/dat[a/jy/GLIP/DATASET')