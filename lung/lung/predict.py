#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSI预测脚本 - 基于train.py训练的模型

功能特点：
1. 前景分割预筛选 - 只处理前景区域的patch
2. 图像学过滤 - 饱和度、白色区域、对比度过滤
3. 批量预测加速 - batch_size=128
4. 双缩略图输出 - 纯标注 + 原图叠加
5. 【新增】可选的Patch提取功能 - 基于预测结果筛选并保存patch

作者: 自动生成
日期: 2026-01-24
更新: 2026-01-28 - 集成patch提取功能
"""

import warnings
import os
import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict
import json
import pandas as pd
from datetime import datetime
import random

warnings.filterwarnings("ignore")

# TensorFlow 配置
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.keras.backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
set_session(sess)

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import metrics, optimizers

import cv2
import openslide
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from tqdm import tqdm
from skimage.exposure import is_low_contrast
from skimage import filters
from skimage.filters import threshold_multiotsu

# 拼音转换（用于文件夹名缩写）
try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("警告: pypinyin未安装，将使用简化的中文处理方式")

# Root directory
ROOT_DIR = os.path.abspath("/home/deeplearning/data/data2/zyy/test")
sys.path.append(ROOT_DIR)


# ===========================
#  颜色映射 (与patch_segmentation_advanced.py相同)
# ===========================

COLOR_MAP: Dict[int, Dict] = {
    0: dict(mpl_color="c", cv_color=(255, 255, 0), name="Normal_cell"),      # 青色 (BGR)
    1: dict(mpl_color="g", cv_color=(0, 255, 0), name="Nacrotic"),           # 绿色
    2: dict(mpl_color="r", cv_color=(0, 0, 255), name="Tumor_Bed"),          # 红色
    3: dict(mpl_color="b", cv_color=(255, 0, 0), name="Remained_Cancer_Cell"), # 蓝色
}

# 预测类别对应的颜色码（用于patch命名）
# R=红色区域(Tumor_Bed), G=绿色区域(Nacrotic), B=蓝色区域(Remained_Cancer_Cell), C=青色区域(Normal_cell)
CLASS_TO_COLOR_CODE = {0: 'C', 1: 'G', 2: 'R', 3: 'B'}


# ===========================
#  中文文件夹名转拼音缩写
# ===========================

def folder_name_to_abbr(folder_name: str) -> str:
    """
    将文件夹名转换为拼音首字母缩写
    例如: '225111肢端淋巴结' -> '225111zdlbj'
    """
    result = []
    for char in folder_name:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符
            if HAS_PYPINYIN:
                py = pinyin(char, style=Style.FIRST_LETTER)[0][0]
                result.append(py)
            else:
                # 没有pypinyin时跳过中文字符
                pass
        elif char.isalnum():  # 数字和字母保留
            result.append(char)
        # 忽略其他字符（标点等）
    return ''.join(result)


# ===========================
#  COCO数据集生成器（从extract_patches移植）
# ===========================

class COCODatasetGenerator:
    """COCO格式标注生成器"""
    
    def __init__(self, images_dir, output_json_path):
        self.images_dir = images_dir
        self.output_json_path = output_json_path
        self.coco_format = self._init_coco_format()

    def _init_coco_format(self):
        return {
            "info": {
                "description": "Nuclei Dataset from Predicted Patches",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "categories": [
                {"id": 1, "name": "Tumor cell", "supercategory": "cell"},
                {"id": 2, "name": "Stroma cell", "supercategory": "cell"},
                {"id": 3, "name": "Lymphocyte", "supercategory": "cell"},
                {"id": 4, "name": "Histocyte", "supercategory": "cell"},
                {"id": 5, "name": "Vascular entothelial cell", "supercategory": "cell"},
                {"id": 6, "name": "Neutrophil", "supercategory": "cell"},
                {"id": 7, "name": "Eosinophil", "supercategory": "cell"},
                {"id": 8, "name": "Plasma cell", "supercategory": "cell"}
            ],
            "images": [],
            "annotations": []
        }

    def _generate_random_circle(self, img_width, img_height):
        """生成随机圆形标注（占位用）"""
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
        """遍历patches目录生成COCO标注"""
        image_id, annotation_id = 1, 1
        patches_dir = os.path.join(self.images_dir, "patches")
        
        if not os.path.exists(patches_dir):
            print(f"警告: patches目录不存在: {patches_dir}")
            return
        
        for img_filename in sorted(os.listdir(patches_dir)):
            if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(patches_dir, img_filename)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"跳过 {img_filename}: {e}")
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
        print(f"COCO标注已保存: {self.output_json_path}")

# ===========================
#  前景分割函数 (从patch_segmentation_advanced.py复用)
# ===========================

def get_stain_matrix_he():
    """获取H&E染色的标准颜色反卷积矩阵"""
    he_matrix = np.array([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78]
    ])
    return he_matrix


def color_deconvolution(img_rgb, stain_matrix=None):
    """颜色反卷积 - 分离染色通道"""
    if stain_matrix is None:
        stain_matrix = get_stain_matrix_he()
    
    img_float = img_rgb.astype(np.float64) + 1
    img_od = -np.log10(img_float / 256.0)
    
    stain_matrix_inv = np.linalg.pinv(stain_matrix)
    
    h, w, c = img_od.shape
    img_od_flat = img_od.reshape(-1, 3)
    stains_flat = np.dot(img_od_flat, stain_matrix_inv.T)
    stains = stains_flat.reshape(h, w, 3)
    
    return stains


def segment_by_saturation(img_rgb):
    """方法1: HSV饱和度分割"""
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1]
    threshold, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask.astype(np.uint8), threshold


def segment_by_lab(img_rgb):
    """方法2: LAB颜色空间分割"""
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    L_channel = img_lab[:, :, 0]
    A_channel = img_lab[:, :, 1]
    
    L_threshold = filters.threshold_otsu(L_channel)
    L_mask = L_channel < L_threshold + 10
    
    A_threshold = filters.threshold_otsu(A_channel)
    A_mask = A_channel > A_threshold - 5
    
    mask = (L_mask | A_mask).astype(np.uint8) * 255
    return mask, (L_threshold, A_threshold)


def segment_by_grayscale(img_rgb):
    """方法3: 灰度阈值分割"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    try:
        thresholds = threshold_multiotsu(gray, classes=3)
        mask = (gray < thresholds[-1]).astype(np.uint8) * 255
    except:
        threshold, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    
    return mask


def segment_by_color_deconv(img_rgb):
    """方法4: 颜色反卷积分割"""
    stains = color_deconvolution(img_rgb)
    
    h_channel = stains[:, :, 0]
    e_channel = stains[:, :, 1]
    
    combined = h_channel + e_channel
    combined = np.clip(combined, 0, None)
    combined = (combined / combined.max() * 255).astype(np.uint8)
    
    threshold, mask = cv2.threshold(
        combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return mask, threshold


def segment_by_texture(img_rgb):
    """方法5: 纹理分析"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    kernel_size = 15
    local_mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
    local_sqr_mean = cv2.blur(gray.astype(np.float64)**2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sqr_mean - local_mean**2, 0))
    
    local_std = (local_std / local_std.max() * 255).astype(np.uint8)
    
    threshold, mask = cv2.threshold(
        local_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return mask


def remove_small_objects(mask, min_area):
    """去除小的连通区域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            result[labels == i] = 255
    
    return result


def fill_small_holes(mask, max_hole_area):
    """填充小孔洞"""
    inverted = 255 - mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    
    result = mask.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max_hole_area:
            result[labels == i] = 255
    
    return result


def segment_foreground_advanced(thumbnail, method='ensemble'):
    """
    高精度前景分割 - 多方法融合
    
    参数:
        thumbnail: PIL Image对象或numpy数组 (RGB格式)
        method: 分割方法 'ensemble', 'saturation', 'lab', 'deconv', 'texture'
        
    返回:
        前景mask (numpy数组, 0=背景, 255=前景)
        方法信息字典
    """
    if isinstance(thumbnail, Image.Image):
        img_rgb = np.array(thumbnail)
    else:
        img_rgb = thumbnail
    
    info = {}
    
    if method == 'ensemble':
        # 多方法融合
        mask_sat, thresh_sat = segment_by_saturation(img_rgb)
        mask_lab, thresh_lab = segment_by_lab(img_rgb)
        mask_gray = segment_by_grayscale(img_rgb)
        mask_deconv, thresh_deconv = segment_by_color_deconv(img_rgb)
        mask_texture = segment_by_texture(img_rgb)
        
        # 加权投票
        masks = [
            (mask_sat > 0).astype(np.float32),
            (mask_lab > 0).astype(np.float32),
            (mask_gray > 0).astype(np.float32),
            (mask_deconv > 0).astype(np.float32),
            (mask_texture > 0).astype(np.float32),
        ]
        
        weights = [1.0, 1.2, 0.8, 1.5, 0.5]
        
        weighted_sum = np.zeros_like(masks[0])
        for m, w in zip(masks, weights):
            weighted_sum += m * w
        
        total_weight = sum(weights)
        final_mask = (weighted_sum >= total_weight * 0.4).astype(np.uint8) * 255
        info['method'] = 'ensemble (5 methods)'
        
    elif method == 'saturation':
        final_mask, thresh = segment_by_saturation(img_rgb)
        info['method'] = 'HSV saturation'
        
    elif method == 'lab':
        final_mask, thresh = segment_by_lab(img_rgb)
        info['method'] = 'LAB color space'
        
    elif method == 'deconv':
        final_mask, thresh = segment_by_color_deconv(img_rgb)
        info['method'] = 'color deconvolution'
        
    elif method == 'texture':
        final_mask = segment_by_texture(img_rgb)
        info['method'] = 'texture analysis'
        
    else:
        raise ValueError(f"未知方法: {method}")
    
    # 形态学后处理
    min_area = final_mask.size * 0.0001
    final_mask = remove_small_objects(final_mask, min_area)
    final_mask = fill_small_holes(final_mask, min_area * 10)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_large)
    
    return final_mask, info


# ===========================
#  图像学过滤函数
# ===========================

def is_valid_patch(patch_np: np.ndarray) -> bool:
    """
    图像学过滤 - 检查patch是否有效
    复用patch_segmentation_advanced.py L750-L773的逻辑
    
    返回:
        True: 有效patch, 应该进行预测
        False: 无效patch, 应该跳过
    """
    # 检查透明像素
    if patch_np.shape[2] == 4:
        if np.any(patch_np[..., 3] == 0):
            return False
        patch_rgb = patch_np[..., :3]
    else:
        patch_rgb = patch_np
    
    patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
    
    # 过滤低饱和度
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    if np.max(s) < 40:
        return False
    
    # 过滤白色区域
    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    white_ratio = np.mean(patch_gray > 240)
    if white_ratio > 0.85:
        return False
    
    # 过滤低对比度
    if is_low_contrast(patch_rgb):
        return False
    
    return True


# ===========================
#  形态学后处理 - 去除孤立点
# ===========================

def morphological_denoise(predictions: Dict[Tuple[int, int], int], num_classes: int = 4,
                          min_region_size: int = 5, iterations: int = 3) -> Dict[Tuple[int, int], int]:
    """
    增强的形态学后处理 - 多轮邻域去噪 + 连通分量过滤
    
    参数:
        predictions: 预测结果字典 {(ix, iy): pred_class}
        num_classes: 类别数量
        min_region_size: 最小区域大小，小于此值的连通分量会被替换
        iterations: 邻域去噪迭代次数
        
    返回:
        去噪后的预测结果字典
    """
    if not predictions:
        return predictions
    
    # 8邻域偏移
    neighbors_offset = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    denoised = predictions.copy()
    total_changed = 0
    
    # ========== 阶段1：多轮邻域去噪 ==========
    print(f"  邻域去噪迭代 {iterations} 轮...")
    for iteration in range(iterations):
        changed_count = 0
        new_denoised = denoised.copy()
        
        for (ix, iy), current_class in denoised.items():
            # 统计周围8个邻居的类别
            neighbor_classes = []
            for dx, dy in neighbors_offset:
                nx, ny = ix + dx, iy + dy
                if (nx, ny) in denoised:
                    neighbor_classes.append(denoised[(nx, ny)])
            
            # 如果邻居数量少于3个，跳过（边缘区域）
            if len(neighbor_classes) < 3:
                continue
            
            # 统计每个类别的出现次数
            class_counts = np.zeros(num_classes, dtype=np.int32)
            for nc in neighbor_classes:
                class_counts[nc] += 1
            
            # 找到邻居中最常见的类别
            most_common_class = np.argmax(class_counts)
            most_common_count = class_counts[most_common_class]
            
            # 放宽条件：当前类别在邻居中少于2次，且最常见类别占多数
            current_in_neighbors = class_counts[current_class]
            if current_in_neighbors <= 1 and most_common_count >= len(neighbor_classes) * 0.6:
                new_denoised[(ix, iy)] = most_common_class
                changed_count += 1
        
        denoised = new_denoised
        total_changed += changed_count
        
        if changed_count == 0:
            print(f"    第{iteration+1}轮: 无变化，提前结束")
            break
        else:
            print(f"    第{iteration+1}轮: 修改 {changed_count} 个点")
    
    # ========== 阶段2：连通分量过滤 ==========
    print(f"  连通分量过滤 (min_size={min_region_size})...")
    
    # 获取坐标范围
    all_coords = list(denoised.keys())
    if not all_coords:
        return denoised
    
    ix_list = [c[0] for c in all_coords]
    iy_list = [c[1] for c in all_coords]
    min_ix, max_ix = min(ix_list), max(ix_list)
    min_iy, max_iy = min(iy_list), max(iy_list)
    
    # 为每个类别分别进行连通分量分析
    component_changed = 0
    
    for target_class in range(num_classes):
        # 创建该类别的二值图像
        width = max_ix - min_ix + 1
        height = max_iy - min_iy + 1
        binary_map = np.zeros((height, width), dtype=np.uint8)
        
        for (ix, iy), pred_class in denoised.items():
            if pred_class == target_class:
                binary_map[iy - min_iy, ix - min_ix] = 255
        
        # 查找连通分量
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        
        # 过滤小的连通分量
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_region_size:
                # 找到这些小区域的坐标
                small_region_coords = np.where(labels == label_id)
                
                for py, px in zip(small_region_coords[0], small_region_coords[1]):
                    orig_ix = px + min_ix
                    orig_iy = py + min_iy
                    
                    if (orig_ix, orig_iy) in denoised:
                        # 查找周围最常见的其他类别
                        neighbor_classes = []
                        for dx, dy in neighbors_offset:
                            nx, ny = orig_ix + dx, orig_iy + dy
                            if (nx, ny) in denoised and denoised[(nx, ny)] != target_class:
                                neighbor_classes.append(denoised[(nx, ny)])
                        
                        if neighbor_classes:
                            # 替换为邻居中最常见的类别
                            class_counts = np.zeros(num_classes, dtype=np.int32)
                            for nc in neighbor_classes:
                                class_counts[nc] += 1
                            denoised[(orig_ix, orig_iy)] = np.argmax(class_counts)
                            component_changed += 1
    
    print(f"  连通分量过滤: 修改 {component_changed} 个点")
    print(f"  总计修改: {total_changed + component_changed} 个点")
    
    return denoised


# ===========================
#  模型构建
# ===========================

def build_model(weights_path: str = None):
    """
    加载完整模型
    train.py 使用 model.save() 保存的是完整模型，需要用 load_model 加载
    """
    from tensorflow.keras.models import load_model
    from tensorflow.keras import backend as K
    
    # 定义自定义评估指标（与train.py中的相同）
    def metric_precision(y_true, y_pred):
        TP = tf.reduce_sum(y_true * tf.round(y_pred))
        FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
        precision = TP / (TP + FP + K.epsilon())
        return precision

    def metric_recall(y_true, y_pred):
        TP = tf.reduce_sum(y_true * tf.round(y_pred))
        FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
        recall = TP / (TP + FN + K.epsilon())
        return recall

    def metric_F1score(y_true, y_pred):
        TP = tf.reduce_sum(y_true * tf.round(y_pred))
        FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
        FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
        precision = TP / (TP + FP + K.epsilon())
        recall = TP / (TP + FN + K.epsilon())
        F1score = 2 * precision * recall / (precision + recall + K.epsilon())
        return F1score
    
    print("加载模型...")
    
    if weights_path and os.path.exists(weights_path):
        print(f"加载完整模型: {weights_path}")
        # 使用 load_model 加载完整模型，需要提供自定义对象
        custom_objects = {
            'metric_precision': metric_precision,
            'metric_recall': metric_recall,
            'metric_F1score': metric_F1score
        }
        model = load_model(weights_path, custom_objects=custom_objects)
    else:
        print(f"警告: 模型文件不存在: {weights_path}，创建新模型...")
        # 如果没有权重文件，创建新模型
        model0 = InceptionResNetV2(
            weights='inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False
        )
        x = model0.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(4, activation='softmax')(x)  # 4个类别
        model = Model(inputs=model0.input, outputs=predictions)
        
        sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    return model


# ===========================
#  WSI预测主函数
# ===========================

def predict_wsi(
    wsi_path: str,
    model,
    output_dir: str,
    patch_size: int = 224,
    batch_size: int = 1024,
    thumb_height: int = 2000,
    fg_method: str = 'ensemble',
    relative_path: str = '',
    min_region_size: int = 5,
    denoise_iterations: int = 3,
    disable_thumbnails: bool = False
):
    """
    WSI预测主函数
    
    参数:
        wsi_path: WSI文件路径
        model: 已加载的模型
        output_dir: 输出目录
        patch_size: patch大小
        batch_size: 批量预测大小
        thumb_height: 缩略图高度
        fg_method: 前景分割方法
        relative_path: 相对于输入目录的路径，用于保持输出目录结构
        min_region_size: 连通分量最小区域大小
        denoise_iterations: 邻域去噪迭代次数
        disable_thumbnails: 是否禁用缩略图生成
    """
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    print(f"\n{'='*60}")
    print(f"处理WSI: {wsi_name}")
    print(f"{'='*60}")
    
    # 1. 打开WSI
    try:
        slide = openslide.open_slide(wsi_path)
    except Exception as e:
        print(f"无法打开WSI文件: {e}")
        return None
    
    w, h = slide.level_dimensions[0]
    print(f"WSI尺寸: {w} x {h}")
    
    # 2. 生成缩略图
    downscale = h / thumb_height if thumb_height > 0 else 1.0
    downscale = max(downscale, 1.0)
    
    thumb_w = int(w / downscale)
    thumb_h = int(h / downscale)
    
    print(f"缩略图尺寸: {thumb_w} x {thumb_h} (downscale={downscale:.2f})")
    
    try:
        thumb = slide.get_thumbnail((thumb_w, thumb_h))
        thumb_np = np.array(thumb)
    except Exception as e:
        print(f"获取缩略图失败: {e}")
        slide.close()
        return None
    
    # 3. 前景分割
    print(f"进行前景分割 (方法: {fg_method})...")
    foreground_mask, fg_info = segment_foreground_advanced(thumb_np, method=fg_method)
    foreground_ratio = np.sum(foreground_mask > 0) / foreground_mask.size * 100
    print(f"  前景占比: {foreground_ratio:.2f}%")
    print(f"  使用方法: {fg_info.get('method', fg_method)}")
    
    # 4. 计算patch坐标和前景预筛选
    num_w = w // patch_size
    num_h = h // patch_size
    total_patches = num_w * num_h
    
    print(f"\n总patch数: {num_w} x {num_h} = {total_patches}")
    
    # 预筛选前景区域的patch坐标
    valid_coords = []
    mask_patch_w = int(max(1, patch_size / downscale))
    mask_patch_h = int(max(1, patch_size / downscale))
    
    for ix in range(num_w):
        for iy in range(num_h):
            x0 = ix * patch_size
            y0 = iy * patch_size
            
            # 映射到缩略图坐标
            mx = int(x0 / downscale)
            my = int(y0 / downscale)
            
            # 检查前景mask
            mx2 = min(mx + mask_patch_w, foreground_mask.shape[1])
            my2 = min(my + mask_patch_h, foreground_mask.shape[0])
            
            if mx < mx2 and my < my2:
                region = foreground_mask[my:my2, mx:mx2]
                if region.size > 0:
                    foreground_ratio_patch = np.mean(region > 0)
                    if foreground_ratio_patch >= 0.5:  # 超过50%是前景
                        valid_coords.append((ix, iy, x0, y0))
    
    print(f"前景区域patch数: {len(valid_coords)} ({100*len(valid_coords)/total_patches:.1f}%)")
    
    # 5. 批量预测
    print(f"\n开始批量预测 (batch_size={batch_size})...")
    
    # 存储预测结果
    predictions = {}  # (ix, iy) -> pred_class
    class_counters = np.zeros(4, dtype=np.int32)
    
    # 批量收集
    batch_patches = []
    batch_coords = []
    
    for ix, iy, x0, y0 in tqdm(valid_coords, desc="读取patch"):
        try:
            region = slide.read_region((x0, y0), 0, (patch_size, patch_size))
            patch_np = np.array(region)
            
            # 图像学过滤
            if not is_valid_patch(patch_np):
                continue
            
            # 转为RGB (去除alpha通道)
            if patch_np.shape[2] == 4:
                patch_rgb = patch_np[..., :3]
            else:
                patch_rgb = patch_np
            
            batch_patches.append(patch_rgb)
            batch_coords.append((ix, iy))
            
            # 达到batch_size时进行预测
            if len(batch_patches) >= batch_size:
                batch_array = np.array(batch_patches)
                preds = model.predict(batch_array, verbose=0)
                pred_classes = np.argmax(preds, axis=1)
                
                for (bix, biy), pred_class in zip(batch_coords, pred_classes):
                    predictions[(bix, biy)] = pred_class
                    class_counters[pred_class] += 1
                
                batch_patches = []
                batch_coords = []
                
        except Exception as e:
            continue
    
    # 处理剩余的patch
    if batch_patches:
        batch_array = np.array(batch_patches)
        preds = model.predict(batch_array, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        
        for (bix, biy), pred_class in zip(batch_coords, pred_classes):
            predictions[(bix, biy)] = pred_class
            class_counters[pred_class] += 1
    
    slide.close()
    
    # 6. 形态学后处理 - 去除孤立点
    print(f"\n进行形态学去噪 (min_region={min_region_size}, iterations={denoise_iterations})...")
    predictions = morphological_denoise(predictions, num_classes=4, 
                                         min_region_size=min_region_size,
                                         iterations=denoise_iterations)
    
    # 重新统计各类别数量（去噪后）
    class_counters = np.zeros(4, dtype=np.int32)
    for pred_class in predictions.values():
        class_counters[pred_class] += 1
    
    # 7. 计算MPR指标
    # MPR = Remained_Cancer_Cell / (Nacrotic + Tumor_Bed + Remained_Cancer_Cell)
    g = class_counters[1]  # Nacrotic (绿色)
    r = class_counters[2]  # Tumor_Bed (红色)
    b = class_counters[3]  # Remained_Cancer_Cell (蓝色)
    
    try:
        mpr = b / (r + g + b)
    except ZeroDivisionError:
        mpr = 0.0
    
    print(f"\n{'='*40}")
    print(f"预测统计:")
    for idx in range(4):
        print(f"  类别 {idx} ({COLOR_MAP[idx]['name']}): {class_counters[idx]}")
    print(f"\nMPR = {mpr:.4f}")
    print(f"  Nacrotic(绿): {g}, Tumor_Bed(红): {r}, Remained_Cancer_Cell(蓝): {b}")
    print(f"{'='*40}")
    
    # 7. 生成双缩略图
    if not disable_thumbnails:
        generate_thumbnails(
            thumb_np, predictions, 
            num_w, num_h, downscale, patch_size,
            wsi_name, mpr, output_dir, relative_path
        )
    else:
        print("已禁用缩略图生成，跳过。")
    
    return {
        'wsi_name': wsi_name,
        'predictions': predictions,
        'class_counters': class_counters,
        'mpr': mpr
    }


def generate_thumbnails(
    thumb_np: np.ndarray,
    predictions: Dict[Tuple[int, int], int],
    num_w: int, num_h: int,
    downscale: float, patch_size: int,
    wsi_name: str, mpr: float,
    output_dir: str,
    relative_path: str = ''
):
    """
    生成双缩略图
    
    1. 纯标注缩略图 - 只显示分类颜色框 (保存到 output_dir/annotation/relative_path/)
    2. 叠加缩略图 - 原图 + 半透明颜色覆盖 (保存到 output_dir/overlay/relative_path/)
    """
    # 创建分类输出目录
    annotation_dir = os.path.join(output_dir, 'annotation', relative_path)
    overlay_dir = os.path.join(output_dir, 'overlay', relative_path)
    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    thumb_h, thumb_w = thumb_np.shape[:2]
    
    # ========== 预计算坐标边界表，避免浮点舍入误差导致的条纹 ==========
    # 关键：预计算所有patch索引对应的起始和结束像素坐标
    # 这样相邻patch共享同一个边界值，不会出现间隙
    x_coords = np.linspace(0, thumb_w, num_w + 1).astype(np.int32)
    y_coords = np.linspace(0, thumb_h, num_h + 1).astype(np.int32)
    
    # ========== 缩略图1: 纯标注 ==========
    print("\n生成纯标注缩略图...")
    fig1 = plt.figure(figsize=(thumb_w / 300, thumb_h / 300), dpi=300)
    ax1 = fig1.add_axes([0, 0, 1, 1])
    ax1.axis("off")
    
    # 白色背景
    white_bg = np.ones((thumb_h, thumb_w, 3), dtype=np.uint8) * 255
    ax1.imshow(white_bg)
    
    # 画矩形框 - 使用预计算坐标
    for (ix, iy), pred_class in predictions.items():
        if ix >= num_w or iy >= num_h:
            continue
        x_start = x_coords[ix]
        x_end = x_coords[ix + 1]
        y_start = y_coords[iy]
        y_end = y_coords[iy + 1]
        
        color = COLOR_MAP[pred_class]['mpl_color']
        rect = plt.Rectangle(
            xy=(x_start, y_start),
            width=x_end - x_start,
            height=y_end - y_start,
            edgecolor=color,
            facecolor=color,
            alpha=0.7,
            linewidth=0
        )
        ax1.add_patch(rect)
    
    annotation_path = os.path.join(annotation_dir, f"{wsi_name}_mpr{mpr:.4f}.png")
    fig1.savefig(annotation_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig1)
    print(f"保存: {annotation_path}")
    
    # ========== 缩略图2: 原图叠加 ==========
    print("生成原图叠加缩略图...")
    fig2 = plt.figure(figsize=(thumb_w / 300, thumb_h / 300), dpi=300)
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.axis("off")
    
    # 显示原图
    ax2.imshow(thumb_np)
    
    # 创建半透明覆盖层 - 使用预计算坐标
    overlay_rgba = np.zeros((thumb_h, thumb_w, 4), dtype=np.float32)
    
    for (ix, iy), pred_class in predictions.items():
        if ix >= num_w or iy >= num_h:
            continue
        mx = x_coords[ix]
        mx2 = x_coords[ix + 1]
        my = y_coords[iy]
        my2 = y_coords[iy + 1]
        
        rgb = mcolors.to_rgb(COLOR_MAP[pred_class]['mpl_color'])
        overlay_rgba[my:my2, mx:mx2, 0] = rgb[0]
        overlay_rgba[my:my2, mx:mx2, 1] = rgb[1]
        overlay_rgba[my:my2, mx:mx2, 2] = rgb[2]
        overlay_rgba[my:my2, mx:mx2, 3] = 0.4  # 半透明
    
    ax2.imshow(overlay_rgba, interpolation='nearest')
    
    overlay_path = os.path.join(overlay_dir, f"{wsi_name}_mpr{mpr:.4f}.png")
    fig2.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    print(f"保存: {overlay_path}")


# ===========================
#  Patch提取功能（从extract_patches_local_multithread.py移植并优化）
# ===========================

def get_dilated_search_mask(
    predictions: Dict[Tuple[int, int], int],
    target_class: int,
    k: int,
    num_w: int, 
    num_h: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    获取基于目标区域膨胀的搜索Mask（轮廓膨胀逻辑）
    
    返回:
        dilated_mask: 膨胀后的二值Mask (0或255)
        dilated_contour: 膨胀后的轮廓点 (用于可视化)
        target_info: 原始目标区域信息
    """
    # 1. 创建目标类别二值图
    binary_mask = np.zeros((num_h, num_w), dtype=np.uint8)
    target_points = []
    
    for (ix, iy), pred_class in predictions.items():
        if pred_class == target_class and ix < num_w and iy < num_h:
            binary_mask[iy, ix] = 255
            target_points.append((ix, iy))
            
    if not target_points:
        return None, None, {'area': 0, 'center': None}

    # 2. 查找最大连通分量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None, None, {'area': 0, 'center': None}
        
    max_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for label_id in range(2, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = label_id
            
    largest_mask = np.zeros_like(binary_mask)
    largest_mask[labels == max_label] = 255
    
    info = {
        'area': max_area,
        'center': (int(centroids[max_label][0]), int(centroids[max_label][1])),
        'bbox': stats[max_label, :4] # x, y, w, h
    }

    # 3. 轮廓膨胀
    kernel_size = 2 * k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(largest_mask, kernel, iterations=1)
    
    # 获取膨胀后的轮廓用于可视化
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dilated_contour = contours[0] if contours else None
    
    return dilated_mask, dilated_contour, info


def create_annotation_canvas(
    predictions: Dict[Tuple[int, int], int],
    num_w: int, 
    num_h: int,
    thumb_w: int,
    thumb_h: int
) -> np.ndarray:
    """
    创建纯标注底图（Annotation Map）
    """
    # 预计算坐标（与generate_thumbnails保持一致）
    x_coords = np.linspace(0, thumb_w, num_w + 1).astype(np.int32)
    y_coords = np.linspace(0, thumb_h, num_h + 1).astype(np.int32)
    
    # 白色背景 (BGR)
    canvas = np.ones((thumb_h, thumb_w, 3), dtype=np.uint8) * 255
    
    for (ix, iy), pred_class in predictions.items():
        if ix >= num_w or iy >= num_h:
            continue
        
        x_start = x_coords[ix]
        x_end = x_coords[ix + 1]
        y_start = y_coords[iy]
        y_end = y_coords[iy + 1]
        
        # COLOR_MAP中存储的是cv_color (BGR)
        color = COLOR_MAP[pred_class]['cv_color']
        
        # 绘制实心矩形(覆盖白色背景)
        canvas[y_start:y_end, x_start:x_end] = color
        
    return canvas


def extract_patches_from_predictions(
    wsi_path: str,
    predictions: Dict[Tuple[int, int], int],
    patch_output_dir: str,
    parent_folder_name: str,
    wsi_name: str,
    patch_size: int,
    k: int,
    max_patches: int,
    num_w: int,
    num_h: int,
    thumb_w: int,    # 新增
    thumb_h: int,    # 新增
    downscale: float # 新增
) -> List[Dict]:
    """
    基于预测结果提取并保存patch (改进版: 轮廓膨胀 + 自定义可视化)
    """
    print(f"\n{'='*40}")
    print(f"开始提取Patch (轮廓膨胀 k={k})...")
    print(f"{'='*40}")
    
    if not predictions:
        print("  警告: 无预测结果，跳过提取")
        return []
    
    # 1. 确定搜索区域 (轮廓膨胀逻辑)
    print("  计算搜索区域...")
    # 优先蓝色区域 (Class 3)
    search_mask, search_contour, target_info = get_dilated_search_mask(
        predictions, target_class=3, k=k, num_w=num_w, num_h=num_h
    )
    region_type = "Non-blue area around tumor"
    
    # 如果没找到蓝色，找红色区域 (Class 2)
    if search_mask is None:
        print("  未找到蓝色区域，尝试红色区域...")
        search_mask, search_contour, target_info = get_dilated_search_mask(
            predictions, target_class=2, k=k, num_w=num_w, num_h=num_h
        )
        region_type = "Area around tumor bed"
        
    # 如果还没找到，使用中心点 (简单矩形逻辑兜底)
    # 这里为了简化，如果没有红蓝区域，我们可以选择不提取或者提取所有
    # 为保持一致性，如果没找到特定区域，我们选择预测中心作为参考点（不使用轮廓膨胀）
    reference_point = target_info.get('center')
    
    if search_mask is None:
        print("  未找到目标区域(红/蓝)，使用预测中心兜底...")
        region_type = "Area around prediction center"
        all_coords = list(predictions.keys())
        if all_coords:
            ix_list = [c[0] for c in all_coords]
            iy_list = [c[1] for c in all_coords]
            cx, cy = sum(ix_list)//len(ix_list), sum(iy_list)//len(iy_list)
            reference_point = (cx, cy)
            # 简单的矩形Mask
            search_mask = np.zeros((num_h, num_w), dtype=np.uint8)
            cv2.rectangle(search_mask, (max(0, cx-k), max(0, cy-k)), (min(num_w, cx+k), min(num_h, cy+k)), 255, -1)
            search_contour = None
        else:
            return []
            
    print(f"  目标类型: {region_type}, 中心: {reference_point}")

    # 2. 生成候选Patch列表 (在search_mask内的前景点)
    candidate_patches = []
    ref_ix, ref_iy = reference_point if reference_point else (num_w//2, num_h//2)
    
    rows, cols = np.where(search_mask > 0)
    for iy, ix in zip(rows, cols):
        if (ix, iy) in predictions:
            # 计算距离用于排序
            dist = abs(ix - ref_ix) + abs(iy - ref_iy)
            candidate_patches.append((dist, ix, iy))
            
    candidate_patches.sort()
    print(f"  搜索范围内候选Patch数: {len(candidate_patches)}")
    
    # 3. 准备可视化
    # 创建Annotation Map作为底图
    vis_img = create_annotation_canvas(predictions, num_w, num_h, thumb_w, thumb_h)
    
    # 绘制黄色搜索轮廓 (需映射到缩略图坐标)
    if search_contour is not None:
        # contour点是 (ix, iy)，映射逻辑: x = ix * patch_size / downscale
        contour_thumb = []
        for point in search_contour:
            px, py = point[0]
            tx = int(px * patch_size / downscale)
            ty = int(py * patch_size / downscale)
            contour_thumb.append([tx, ty])
        
        if contour_thumb:
            pts = np.array(contour_thumb, np.int32).reshape((-1, 1, 2))
            # 黄色 (0, 255, 255) BGR
            cv2.polylines(vis_img, [pts], True, (0, 255, 255), 2)
            
    # 4. 提取与保存
    # 建立保存目录
    folder_abbr = folder_name_to_abbr(parent_folder_name)
    patches_dir = os.path.join(patch_output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    
    try:
        slide = openslide.open_slide(wsi_path)
    except:
        return []
        
    patch_results = []
    saved_count = 0
    saved_patches_for_vis = [] # 用于最后可视化绘制
    
    # 计算目标区域在WSI层面的面积和周长 (大概估算)
    target_area_wsi = target_info['area'] * (patch_size ** 2)
    # 周长估算较复杂，这里简化处理，如有轮廓可用轮廓长
    target_perimeter_wsi = 0 
    if search_contour is not None:
         target_perimeter_wsi = cv2.arcLength(search_contour, True) * patch_size

    for dist, ix, iy in tqdm(candidate_patches, desc="  提取Patch"):
        if saved_count >= max_patches:
            break
            
        pred_class = predictions[(ix, iy)]
        color_code = CLASS_TO_COLOR_CODE.get(pred_class, 'U')
        
        # 过滤：只保存红(2)和绿(1)
        if pred_class not in [1, 2]:
            continue
            
        # 读取图像
        x0, y0 = ix * patch_size, iy * patch_size
        try:
            region = slide.read_region((x0, y0), 0, (patch_size, patch_size))
            patch_bgr = cv2.cvtColor(np.array(region), cv2.COLOR_RGBA2BGR)
        except:
            continue
            
        # 计算向量 (简化版: 此时已有reference_point)
        # 如果需要更精确的"最近轮廓点向量"，需复用extract_patches_local_multithread中的get_closest_contour_point
        # 这里使用到中心点的向量近似
        vx_wsi = (ref_ix - ix) * patch_size
        vy_wsi = (ref_iy - iy) * patch_size
        
        # 保存文件
        fname = (f"{folder_abbr}-{wsi_name}_patch_x{ix}_y{iy}_color{color_code}_"
                 f"TargetTumorArea{int(target_area_wsi)}_Perimeter{int(target_perimeter_wsi)}_"
                 f"vector2targetbyVX{vx_wsi}VY{vy_wsi}.png")
        cv2.imwrite(os.path.join(patches_dir, fname), patch_bgr)
        
        saved_count += 1
        patch_results.append({
            "parent_folder": parent_folder_name,
            "folder_abbr": folder_abbr,
            "wsi_id": wsi_name,
            "patch_x": ix, "patch_y": iy,
            "patch_filename": fname,
            "patch_color": color_code,
            "pred_class": pred_class,
            "distance_to_reference": dist,
            "region_type": region_type
        })
        
        # 记录用于可视化
        saved_patches_for_vis.append({
            'ix': ix, 'iy': iy, 'color_code': color_code
        })
        
    slide.close()
    print(f"  保存了 {saved_count} 个Patch")
    
    # 5. 完成可视化绘制并保存
    # 绘制已保存Patch的填充矩形
    # 预计算坐标网格加速绘制
    x_grid = np.linspace(0, thumb_w, num_w + 1).astype(np.int32)
    y_grid = np.linspace(0, thumb_h, num_h + 1).astype(np.int32)
    
    for p in saved_patches_for_vis:
        ix, iy = p['ix'], p['iy']
        if ix >= num_w or iy >= num_h: continue
        
        x_s, x_e = x_grid[ix], x_grid[ix+1]
        y_s, y_e = y_grid[iy], y_grid[iy+1]
        
        # 颜色: R(Class2)->紫色(255, 0, 255), G(Class1)->粉色(203, 192, 255) BGR
        if p['color_code'] == 'R':
            draw_color = (255, 0, 255) # 紫色
        else:
            draw_color = (203, 192, 255) # 粉色
            
        # 绘制实心矩形
        cv2.rectangle(vis_img, (x_s, y_s), (x_e, y_e), draw_color, -1)

    vis_save_dir = os.path.join(patch_output_dir, "visualizations", "".join(parent_folder_name.split()) ) # 简单处理路径
    # 这里为了保持原有结构，我们尽量用relativePath如果传入了的话，但这里参数里没传relative_path，暂用parent_folder名
    # 更好的方式是沿用调用者的目录结构
    vis_save_dir = os.path.join(patch_output_dir, "visualizations")
    os.makedirs(vis_save_dir, exist_ok=True)
    vis_path = os.path.join(vis_save_dir, f"{wsi_name}_extraction_vis.png")
    cv2.imwrite(vis_path, vis_img)
    print(f"  可视化图已保存: {vis_path}")
    
    return patch_results


# ===========================
#  主函数
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description='WSI预测脚本 - 基于train.py训练的模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 只预测
  python predict.py --wsi /path/to/slide.mrxs --weights /path/to/model.h5
  python predict.py --wsi_dir /path/to/wsi_folder/ --weights /path/to/model.h5

  # 预测 + 提取patch
  python predict.py --wsi_dir /path/to/wsi_folder/ --extract_patches --patch_output_dir ./extracted_patches/
'''
    )
    
    parser.add_argument(
        '--wsi', '-w',
        default=None,
        help='单个WSI文件路径'
    )
    
    parser.add_argument(
        '--wsi_dir', '-d',
        default=None,
        help='WSI文件夹路径（批量处理）'
    )
    
    parser.add_argument(
        '--weights',
        default='/home/deeplearning/data/data2/zyy/test/weight/LUNG_NEW_final.h5',
        help='模型权重路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='/home/deeplearning/data/data2/zyy/test/output_predict/',
        help='预测输出目录（缩略图保存位置）'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='批量预测大小 (默认1024)'
    )
    
    parser.add_argument(
        '--patch_size',
        type=int,
        default=224,
        help='Patch大小 (默认224)'
    )
    
    parser.add_argument(
        '--thumb_height',
        type=int,
        default=2000,
        help='缩略图高度 (默认2000)'
    )
    
    parser.add_argument(
        '--fg_method',
        choices=['ensemble', 'saturation', 'lab', 'deconv', 'texture'],
        default='ensemble',
        help='前景分割方法 (默认ensemble)'
    )
    
    parser.add_argument(
        '--min_region_size',
        type=int,
        default=5,
        help='连通分量最小区域大小，小于此值的区域会被过滤 (默认5)'
    )
    
    parser.add_argument(
        '--denoise_iterations',
        type=int,
        default=3,
        help='邻域去噪迭代次数 (默认3)'
    )
    
    # ========== Patch提取相关参数 ==========
    parser.add_argument(
        '--extract_patches',
        action='store_true',
        help='启用patch提取功能'
    )
    
    parser.add_argument(
        '--patch_output_dir',
        default='/home/deeplearning/data/data2/zyy/test/extracted_patches/',
        help='patch提取输出目录'
    )
    
    parser.add_argument(
        '--extract_k',
        type=int,
        default=5,
        help='patch提取搜索范围扩展倍数 (默认5)'
    )
    
    parser.add_argument(
        '--max_patches',
        type=int,
        default=1000000,
        help='每个WSI最大提取patch数 (默认1000000)'
    )

    parser.add_argument(
        '--no_thumbnails',
        action='store_true',
        help='不生成预测结果缩略图 (annotation和overlay)'
    )
    
    args = parser.parse_args()
    
    # 检查输入
    if args.wsi is None and args.wsi_dir is None:
        print("错误: 请指定 --wsi 或 --wsi_dir")
        return
    
    # 构建模型
    model = build_model(args.weights)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 如果启用patch提取，创建相关目录
    if args.extract_patches:
        os.makedirs(args.patch_output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.patch_output_dir, "patches"), exist_ok=True)
        os.makedirs(os.path.join(args.patch_output_dir, "visualizations"), exist_ok=True)
        all_patch_results = []  # 收集所有patch结果
    
    # 收集WSI文件和相对路径
    wsi_files = []  # [(wsi_path, relative_path, parent_folder_name), ...]
    wsi_base_dir = args.wsi_dir  # 用于计算相对路径
    
    if args.wsi:
        # 单文件模式
        wsi_path = os.path.abspath(args.wsi)
        parent_folder = os.path.basename(os.path.dirname(wsi_path))
        wsi_files.append((wsi_path, '', parent_folder))
    if args.wsi_dir:
        wsi_base_dir = os.path.abspath(args.wsi_dir)
        for root, dirs, files in os.walk(args.wsi_dir, followlinks=True):
            for fname in files:
                if fname.endswith('.mrxs'):
                    wsi_path = os.path.join(root, fname)
                    # 计算相对于输入目录的路径（只取文件夹部分）
                    rel_dir = os.path.relpath(root, wsi_base_dir)
                    if rel_dir == '.':
                        rel_dir = ''
                    # 获取父文件夹名
                    parent_folder = os.path.basename(root)
                    wsi_files.append((wsi_path, rel_dir, parent_folder))
    
    print(f"\n找到 {len(wsi_files)} 个WSI文件")
    if args.extract_patches:
        print(f"已启用Patch提取功能，输出目录: {args.patch_output_dir}")
    
    # 逐个处理
    for idx, (wsi_path, relative_path, parent_folder) in enumerate(wsi_files):
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        print(f"\n[{idx+1}/{len(wsi_files)}] 处理: {os.path.basename(wsi_path)}")
        if relative_path:
            print(f"  相对路径: {relative_path}")
        
        # 执行预测
        result = predict_wsi(
            wsi_path=wsi_path,
            model=model,
            output_dir=args.output,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            thumb_height=args.thumb_height,
            fg_method=args.fg_method,
            relative_path=relative_path,
            min_region_size=args.min_region_size,
            denoise_iterations=args.denoise_iterations,
            disable_thumbnails=args.no_thumbnails
        )
        
        # 如果启用patch提取且预测成功
        if args.extract_patches and result and result.get('predictions'):
            # 获取grid尺寸和缩略图尺寸
            slide = openslide.open_slide(wsi_path)
            w, h = slide.level_dimensions[0]
            
            # 重新计算缩略图参数（需与predict_wsi中一致）
            downscale = h / args.thumb_height if args.thumb_height > 0 else 1.0
            downscale = max(downscale, 1.0)
            thumb_w = int(w / downscale)
            thumb_h = int(h / downscale)
            
            slide.close()
            num_w = w // args.patch_size
            num_h = h // args.patch_size
            
            # 提取patch
            patch_results = extract_patches_from_predictions(
                wsi_path=wsi_path,
                predictions=result['predictions'],
                patch_output_dir=args.patch_output_dir,
                parent_folder_name=parent_folder,
                wsi_name=wsi_name,
                patch_size=args.patch_size,
                k=args.extract_k,
                max_patches=args.max_patches,
                num_w=num_w,
                num_h=num_h,
                thumb_w=thumb_w,    # 新增
                thumb_h=thumb_h,    # 新增
                downscale=downscale # 新增
            )
            
            all_patch_results.extend(patch_results)
            
            # (旧的可视化逻辑已移除，由extract_patches_from_predictions内部自动生成)
            # 只有在生成了缩略图的情况下才复制
            if not args.no_thumbnails:
                vis_dir = os.path.join(args.patch_output_dir, "visualizations", relative_path)
                os.makedirs(vis_dir, exist_ok=True)
                
                # 复制annotation缩略图 (作为备份)
                src_annotation = os.path.join(args.output, 'annotation', relative_path, 
                                              f"{wsi_name}_mpr{result['mpr']:.4f}.png")
                if os.path.exists(src_annotation):
                    import shutil
                    dst_annotation = os.path.join(vis_dir, f"{wsi_name}_mpr{result['mpr']:.4f}_original.png")
                    shutil.copy2(src_annotation, dst_annotation)
    
    # 如果启用patch提取，生成COCO JSON和CSV
    if args.extract_patches:
        print("\n" + "="*60)
        print("生成COCO标注和CSV报表...")
        print("="*60)
        
        # 生成COCO JSON
        coco_json_path = os.path.join(args.patch_output_dir, "coco_annotations.json")
        coco_generator = COCODatasetGenerator(args.patch_output_dir, coco_json_path)
        coco_generator.generate_dataset()
        
        # 生成CSV报表
        if all_patch_results:
            csv_path = os.path.join(args.patch_output_dir, "patch_details.csv")
            df = pd.DataFrame(all_patch_results)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"CSV报表已保存: {csv_path}")
            print(f"总计提取 {len(all_patch_results)} 个patch")
        else:
            print("警告: 未提取到任何patch")
    
    print("\n全部处理完成!")


if __name__ == "__main__":
    main()
