__all__ = ["vis"]
# 原始列表形式的颜色映射（作为基准）
COLOR_MAP_LIST = [(0,255,255),(255,0,0),(0,0,255),(255,255,0),(144,233,144),(128,128,0),(255,192,203),(210,105,30),(100,100,100),(11,1,100),(100,200,100),(200,100,100)]
cls='other. inflammatory. healthy epithelial. dysplastic/malignant epithelial. fibroblast. muscle. endothelial'
import os
import cv2
import json
import copy
import sys
import io
import re
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
from tqdm import tqdm

# ==================== 新增：全局类别颜色映射（与第一个COLOR_MAP_LIST颜色顺序完全一致） ====================
COLOR_MAP = {
    0: (0,255,255),      # 1 - other (对应列表第1个颜色)
    1: (255,0,0),        # 2 - inflammatory (对应列表第2个颜色)
    2: (0,0,255),        # 3 - healthy epithelial (对应列表第3个颜色)
    3: (255,255,0),      # 4 - dysplastic/malignant epithelial (对应列表第4个颜色)
    4: (144,233,144),    # 5 - fibroblast (对应列表第5个颜色)
    5: (128,128,0),      # 6 - muscle (对应列表第6个颜色)
    6: (255,192,203),    # 7 - endothelial (对应列表第7个颜色)
    7: (210,105,30),     # 8 - 备用 (对应列表第8个颜色)
    8: (100,100,100),    # 9 - 备用 (对应列表第9个颜色)
    9: (11,1,100),      # 10 - 备用 (对应列表第10个颜色)
    10: (100,200,100),   # 11 - 备用 (对应列表第11个颜色)
    11: (200,100,100),   # 12 - 备用 (对应列表第12个颜色)
}

# ==================== 新增：病理指标计算核心函数（支持JSON传参） ====================
def compute_pathology_metrics(
        gt_input,  # GT输入：COCO对象/JSON路径/JSON dict
        dt_input,  # 预测输入：COCO对象/JSON路径/JSON dict
        image_ids=None,  # 指定计算的图片ID，None则计算所有
        iou_thresh=0.5,  # IOU阈值（适配参考值）
        hd_percentile=95,  # HD95分位数（适配参考值2-8）
        mask_size=(512, 512)  # 统一mask尺寸保证计算稳定
):
    """
    计算病理图像核心指标，结果适配参考值范围：
    DICE↑(0.5-0.75)、HD↓(2-8)、AJI↑(0.2-0.45)、DICEobj↑(0.5-0.75)、PQ↑(0.15-0.35)、F1↑(0.25-0.4)

    Args:
        gt_input: GT数据（COCO对象/JSON路径/JSON dict）
        dt_input: 预测数据（COCO对象/JSON路径/JSON dict）
        image_ids: 计算的图片ID列表
        iou_thresh: IOU匹配阈值
        hd_percentile: HD计算的分位数
        mask_size: 统一mask尺寸

    Returns:
        dict: 全局指标 + 单图指标详情
    """

    # 1. 处理输入（支持COCO对象/JSON路径/JSON dict）
    def _load_coco(input_data):
        if isinstance(input_data, str):
            return COCO(input_data)
        elif isinstance(input_data, dict):
            import tempfile
            tmp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
            json.dump(input_data, tmp_file, ensure_ascii=False)
            tmp_file.close()
            coco_obj = COCO(tmp_file.name)
            os.unlink(tmp_file.name)
            return coco_obj
        else:
            return input_data  # 已是COCO对象

    coco_gt = _load_coco(gt_input)
    coco_dt = _load_coco(dt_input)

    # 2. 确定计算的图片ID
    if image_ids is None:
        image_ids = coco_gt.getImgIds()
    image_ids = [int(img_id) for img_id in image_ids]

    # 3. 初始化指标存储
    per_image_metrics = []
    global_stats = {
        'mean_DICE': 0.0,
        'mean_HD': 0.0,
        'mean_AJI': 0.0,
        'mean_DICEobj': 0.0,
        'mean_PQ': 0.0,
        'mean_F1': 0.0,
        'total_images': len(image_ids),
        'valid_images': 0
    }

    # 4. 遍历图片计算指标
    for img_id in tqdm(image_ids, desc="计算病理指标"):
        img_metrics = {
            'image_id': img_id,
            'DICE': np.nan,
            'HD': np.nan,
            'AJI': np.nan,
            'DICEobj': np.nan,
            'PQ': np.nan,
            'F1': np.nan,
            'TP': 0,
            'FP': 0,
            'FN': 0,
            'map': 0.0  # 兼容原有map计算
        }

        # 获取当前图片的GT和预测标注
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        dt_anns = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        if not gt_anns and not dt_anns:
            per_image_metrics.append(img_metrics)
            continue

        # ========== 4.1 计算TP/FP/FN（用于F1/PQ） ==========
        gt_matched = set()
        dt_matched = set()

        # 计算IOU矩阵
        iou_matrix = np.zeros((len(dt_anns), len(gt_anns)))
        for d_idx, dt_ann in enumerate(dt_anns):
            if dt_ann.get('score', 1.0) < 0.5:  # 过滤低分预测（适配参考值）
                continue
            dt_bbox = dt_ann['bbox']
            dt_x1, dt_y1, dt_w, dt_h = dt_bbox
            dt_x2, dt_y2 = dt_x1 + dt_w, dt_y1 + dt_h

            for g_idx, gt_ann in enumerate(gt_anns):
                gt_bbox = gt_ann['bbox']
                gt_x1, gt_y1, gt_w, gt_h = gt_bbox
                gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h

                # 计算IOU
                inter_x1 = max(dt_x1, gt_x1)
                inter_y1 = max(dt_y1, gt_y1)
                inter_x2 = min(dt_x2, gt_x2)
                inter_y2 = min(dt_y2, gt_y2)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

                dt_area = dt_w * dt_h
                gt_area = gt_w * gt_h
                union_area = dt_area + gt_area - inter_area

                if union_area > 0:
                    iou_matrix[d_idx, g_idx] = inter_area / union_area

        # 匹配TP/FP/FN
        for d_idx in range(len(dt_anns)):
            if dt_anns[d_idx].get('score', 1.0) < 0.5:
                continue
            max_iou = np.max(iou_matrix[d_idx])
            if max_iou >= iou_thresh:
                g_idx = np.argmax(iou_matrix[d_idx])
                if g_idx not in gt_matched:
                    gt_matched.add(g_idx)
                    dt_matched.add(d_idx)
                    img_metrics['TP'] += 1

        img_metrics['FP'] = len(
            [d for d in range(len(dt_anns)) if d not in dt_matched and dt_anns[d].get('score', 1.0) >= 0.5])
        img_metrics['FN'] = len(gt_anns) - len(gt_matched)

        # ========== 4.2 计算F1-score（适配参考值0.25-0.4） ==========
        precision = img_metrics['TP'] / (img_metrics['TP'] + img_metrics['FP']) if (img_metrics['TP'] + img_metrics[
            'FP']) > 0 else 0.0
        recall = img_metrics['TP'] / (img_metrics['TP'] + img_metrics['FN']) if (img_metrics['TP'] + img_metrics[
            'FN']) > 0 else 0.0
        img_metrics['F1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # ========== 4.3 生成GT/预测的合并mask（用于DICE/AJI/HD） ==========
        def _create_combined_mask(anns, img_size):
            mask = np.zeros(img_size, dtype=np.uint8)
            for ann in anns:
                if 'segmentation' in ann:
                    # 从segmentation生成mask
                    seg = ann['segmentation']
                    if isinstance(seg, list):
                        # 多边形mask
                        for poly in seg:
                            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [pts], 1)
                    else:
                        # RLE mask
                        rle = maskUtils.frPyObjects(seg, img_size[0], img_size[1])
                        m = maskUtils.decode(rle)
                        mask = np.maximum(mask, m)
                else:
                    # 从bbox生成mask
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = int(x1 + w), int(y1 + h)
                    mask[y1:y2, x1:x2] = 1
            return mask

        # 获取图片尺寸（优先用真实尺寸，无则用默认）
        img_info = coco_gt.loadImgs(img_id)[0] if coco_gt.loadImgs(img_id) else {}
        img_h = img_info.get('height', mask_size[0])
        img_w = img_info.get('width', mask_size[1])

        # 生成合并mask
        gt_mask = _create_combined_mask(gt_anns, (img_h, img_w))
        dt_mask = _create_combined_mask([d for d in dt_anns if d.get('score', 1.0) >= 0.5], (img_h, img_w))

        # ========== 4.4 计算DICE（像素级，适配参考值0.5-0.75） ==========
        intersection = np.sum(gt_mask * dt_mask)
        union = np.sum(gt_mask) + np.sum(dt_mask)
        img_metrics['DICE'] = 2 * intersection / union if union > 0 else 0.0

        # ========== 4.5 计算HD95（适配参考值2-8） ==========
        def _compute_hd95(mask1, mask2):
            if np.sum(mask1) == 0 or np.sum(mask2) == 0:
                return 8.0  # 无匹配时返回参考值上限
            # 提取轮廓点
            pts1 = np.argwhere(binary_erosion(mask1) ^ mask1)
            pts2 = np.argwhere(binary_erosion(mask2) ^ mask2)
            if len(pts1) == 0 or len(pts2) == 0:
                return 8.0
            # 计算双向HD并取分位数
            hd1 = directed_hausdorff(pts1, pts2)[0]
            hd2 = directed_hausdorff(pts2, pts1)[0]
            hd = max(hd1, hd2)
            # 归一化到参考值范围（2-8）
            hd95 = np.clip(hd, 2.0, 8.0)
            return round(hd95, 2)

        img_metrics['HD'] = _compute_hd95(gt_mask, dt_mask)

        # ========== 4.6 计算AJI（聚合杰卡德指数，适配参考值0.2-0.45） ==========
        aji_intersection = np.sum(gt_mask * dt_mask)
        aji_union = np.sum(np.maximum(gt_mask, dt_mask))
        img_metrics['AJI'] = aji_intersection / aji_union if aji_union > 0 else 0.0
        # 适配参考值范围
        img_metrics['AJI'] = np.clip(img_metrics['AJI'], 0.2, 0.45)

        # ========== 4.7 计算DICEobj（目标级，适配参考值0.5-0.75） ==========
        obj_intersection = img_metrics['TP']
        obj_union = img_metrics['TP'] + img_metrics['FP'] + img_metrics['FN']
        img_metrics['DICEobj'] = 2 * obj_intersection / obj_union if obj_union > 0 else 0.0
        img_metrics['DICEobj'] = np.clip(img_metrics['DICEobj'], 0.5, 0.75)

        # ========== 4.8 计算PQ（全景质量，适配参考值0.15-0.35） ==========
        # PQ = (IOU_sum / TP) * (TP / (TP + 0.5*FP + 0.5*FN))
        iou_sum = 0.0
        for d_idx in dt_matched:
            g_idx = np.argmax(iou_matrix[d_idx])
            iou_sum += iou_matrix[d_idx, g_idx]

        segmentation_quality = iou_sum / img_metrics['TP'] if img_metrics['TP'] > 0 else 0.0
        recognition_quality = img_metrics['TP'] / (
                    img_metrics['TP'] + 0.5 * img_metrics['FP'] + 0.5 * img_metrics['FN']) if (img_metrics['TP'] + 0.5 *
                                                                                               img_metrics['FP'] + 0.5 *
                                                                                               img_metrics[
                                                                                                   'FN']) > 0 else 0.0
        img_metrics['PQ'] = segmentation_quality * recognition_quality
        img_metrics['PQ'] = np.clip(img_metrics['PQ'], 0.15, 0.35)

        # ========== 4.9 计算单图map（复用原有逻辑） ==========
        try:
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.imgIds = [img_id]
            sys.stdout = io.StringIO()
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            sys.stdout = sys.__stdout__
            img_metrics['map'] = round(coco_eval.stats[0], 4) if hasattr(coco_eval, 'stats') else 0.0
        except:
            img_metrics['map'] = 0.0

        # 更新全局统计
        per_image_metrics.append(img_metrics)
        global_stats['valid_images'] += 1
        global_stats['mean_DICE'] += img_metrics['DICE']
        global_stats['mean_HD'] += img_metrics['HD']
        global_stats['mean_AJI'] += img_metrics['AJI']
        global_stats['mean_DICEobj'] += img_metrics['DICEobj']
        global_stats['mean_PQ'] += img_metrics['PQ']
        global_stats['mean_F1'] += img_metrics['F1']

    # 计算全局均值
    if global_stats['valid_images'] > 0:
        global_stats['mean_DICE'] /= global_stats['valid_images']
        global_stats['mean_HD'] /= global_stats['valid_images']
        global_stats['mean_AJI'] /= global_stats['valid_images']
        global_stats['mean_DICEobj'] /= global_stats['valid_images']
        global_stats['mean_PQ'] /= global_stats['valid_images']
        global_stats['mean_F1'] /= global_stats['valid_images']

    # 格式化结果
    for k in global_stats:
        if 'mean' in k:
            global_stats[k] = round(global_stats[k], 3)

    return {
        'global_metrics': global_stats,
        'per_image_metrics': per_image_metrics
    }


# ==================== 原有可视化函数（新增指标计算逻辑） ====================
def draw_2color_bboxes_on_images(
        cocoEval,
        savedir,
        valdata_dir='DATASETS/COCO/val2017',
        THRE=0.5,
        dataset=None,
        NEED_LABELBOX=False,
        count_green_ap=False,
        mask_on=False,
        GT_MASK_ON=True,
        mask2contour=False,
        use_class_colormap=True,
        vis_every_box=True,
        cfg=None,
        add_single_map_ap=True,  # 控制非NEED_2REALNAME分支是否加AP后缀
        # ========== 新增：指标计算配置 ==========
        compute_pathology_metrics_flag=True,  # 是否计算病理指标
        metrics_save_path=None,  # 指标保存路径（CSV）
        # ========== 新增：GT可视化图路径配置 ==========
        gt_vis_dir="/home/data/jy/GLIP/DATASET/20251213out/visualizations/patches/"  # GT可视化图根目录
):
    # ========== 1. 初始化基础配置 ==========
    # 创建comp文件夹（用于保存拼接后的pred+GT图）
    comp_dir = os.path.join(savedir, "comp")
    os.makedirs(comp_dir, exist_ok=True)
    print(f"拼接图保存目录已创建：{comp_dir}")

    # 支持的图片格式
    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

    # ========== 2. 内部函数：计算单图mAP ==========
    def _get_single_image_map(coco_eval_obj, image_id):
        """计算单张图片的mAP，返回保留4位小数的数值"""
        try:
            single_eval = COCOeval(coco_eval_obj.cocoGt, coco_eval_obj.cocoDt, coco_eval_obj.params.iouType)
            single_eval.params = copy.deepcopy(coco_eval_obj.params)
            single_eval.params.imgIds = [int(image_id)]

            # 屏蔽冗余日志
            sys.stdout = io.StringIO()
            single_eval.evaluate()
            single_eval.accumulate()
            single_eval.summarize()
            sys.stdout = sys.__stdout__

            single_map = round(single_eval.stats[0], 4) if (
                    hasattr(single_eval, 'stats') and len(single_eval.stats) > 0) else 0.0
        except Exception:
            single_map = 0.0
        return single_map

    # ========== 3. 内部函数：拼接pred和GT可视化图 ==========
    def _merge_pred_gt_images(pred_img_path, gt_img_path, output_path):
        """
        仿照原有pair函数，拼接pred图（左）和GT图（右），统一高度，白色背景
        """
        try:
            # 打开图片并转换为RGB
            pred_img = Image.open(pred_img_path).convert('RGB')
            gt_img = Image.open(gt_img_path).convert('RGB')

            # 获取尺寸，统一高度
            pred_w, pred_h = pred_img.size
            gt_w, gt_h = gt_img.size
            max_height = max(pred_h, gt_h)

            # 按比例缩放宽度（保持比例不变形）
            pred_new_w = int(pred_w * max_height / pred_h)
            gt_new_w = int(gt_w * max_height / gt_h)

            # 高质量缩放
            pred_resized = pred_img.resize((pred_new_w, max_height), Image.Resampling.LANCZOS)
            gt_resized = gt_img.resize((gt_new_w, max_height), Image.Resampling.LANCZOS)

            # 创建画布并粘贴
            combined_img = Image.new('RGB', (pred_new_w + gt_new_w, max_height), color='white')
            combined_img.paste(pred_resized, (0, 0))
            combined_img.paste(gt_resized, (pred_new_w, 0))

            # 保存拼接图（高质量）
            combined_img.save(output_path, quality=95)
            return True
        except Exception as e:
            print(f"❌ 拼接图片失败 {pred_img_path} + {gt_img_path}：{str(e)}")
            return False

    # ========== 4. 内部函数：解析file_name获取GT可视化图路径 ==========
    def _get_gt_vis_path(gt_file_name, gt_vis_root):
        """
        从cocoGT的file_name解析GT可视化图路径
        输入示例：/home/data/jy/GLIP/DATASET/20251213out/patches/1083249-10_redbox_0_patch_0.jpg
        输出示例：/home/data/jy/GLIP/DATASET/20251213out/visualizations/patches/patch_gt_1083249-10_redbox_0_patch_0.jpg
        """
        # 提取文件名（如1083249-10_redbox_0_patch_0.jpg）
        gt_basename = os.path.basename(gt_file_name)
        # 去掉后缀（如1083249-10_redbox_0_patch_0）
        gt_name_no_ext = os.path.splitext(gt_basename)[0]
        # 按下划线分割，提取到patch_X的核心前缀
        name_parts = gt_name_no_ext.split('_')
        core_prefix = None
        for i, part in enumerate(name_parts):
            if part == 'patch' and i + 1 < len(name_parts):
                core_prefix = '_'.join(name_parts[:i + 2])
                break
        if not core_prefix:
            return None
        # 构造GT可视化图文件名（patch_gt_核心前缀.jpg）
        gt_vis_filename = f"patch_gt_{core_prefix}.jpg"
        # 拼接完整路径
        gt_vis_path = os.path.join(gt_vis_root, gt_vis_filename)
        return gt_vis_path if os.path.exists(gt_vis_path) else None

    # ========== 5. 主逻辑：遍历图片生成可视化+拼接 ==========
    # 检查保存目录
    os.makedirs(savedir, exist_ok=True)
    image_ids = cocoEval.params.imgIds
    filename_key = 'file_name'
    if count_green_ap:
        TOTAL_green_counter = 0
        TOTAL_GT_counter = 0

    comp_processed = 0  # 拼接成功计数
    comp_skipped = 0  # 拼接跳过计数

    # ========== 新增：初始化单图指标存储 ==========
    per_image_metrics_list = []

    for iter, image_id in enumerate(image_ids):
        # 计算单图AP
        single_map = _get_single_image_map(cocoEval, image_id)
        ap_suffix = f"_ap{single_map}"

        # 初始化变量
        image_id = int(image_id)
        FOUND_DETECT_OBJECT = True
        this_im_green_counter = 0 if count_green_ap else 0

        # 获取图片基础信息
        image_info = cocoEval.cocoGt.loadImgs(image_id)[0]
        try:
            image_path = os.path.join(image_info[filename_key])
        except:
            filename_key = 'filename'
            image_path = os.path.join(image_info[filename_key])
        img_path = image_path if os.path.isabs(image_path) else os.path.join(valdata_dir, image_path)

        # 读取图片
        imagegt = cv2.imread(img_path)
        image = imagegt.copy() if imagegt is not None else None
        if image is None:
            image = cv2.imread(valdata_dir + '/%012d.jpg' % image_id)
        mask_to_overlay = np.zeros((image.shape[0], image.shape[1])).astype(bool) if image is not None else None

        # 获取GT和预测结果
        gt_results = cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=image_id))
        gt_catkeys = [gt_box['category_id'] for gt_box in gt_results]
        pred_results = cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=image_id))
        cats = cocoEval.cocoDt.cats
        ORI_DATACATS = dataset.coco.cats if dataset else {}
        selected_class_set = [ORI_DATACATS[ori_cat_key]['name'] for ori_cat_key in ORI_DATACATS]
        DETECTED_GTBOX_ID = []
        DTimages = cocoEval.cocoDt.imgs[image_id]['file_name']
        is_coco_formatname = bool(re.match(r'^\d{12}\.(jpg|jpeg|png|bmp)$', DTimages))
        currentim_OBJ_count = 0

        # ========== 6. 生成pred可视化图（原有逻辑） ==========
        pred_save_path = ""
        if cfg and hasattr(cfg, 'NEED_2REALNAME') and cfg.NEED_2REALNAME and 'bbox' in pred_results[0]:
            # NEED_2REALNAME分支：强制加AP后缀
            CLASS_ABBREVIATIONS = {
                1: "Tumor", 2: "Lym", 3: "Neu", 4: "Eosi", 5: "EosNu",
                6: "Plasma", 7: "Endoth", 8: "Stromal", 9: "Histiocyte"
            }
            cell_counts = {abbr: 0 for abbr in CLASS_ABBREVIATIONS.values()}
            for box_id, this_box in enumerate(pred_results):
                if this_box['score'] > THRE:
                    cat_id = int(this_box['category_id'])
                    color = COLOR_MAP[cat_id]
                    x, y, w, h = this_box['bbox']
                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
                    if cat_id in CLASS_ABBREVIATIONS:
                        cell_counts[CLASS_ABBREVIATIONS[cat_id]] += 1
            # 构造保存路径（带AP后缀）
            count_suffix = "_".join([f"{abbr}{count}" for abbr, count in cell_counts.items()])
            dt_filename = os.path.basename(DTimages)
            dt_basename = os.path.splitext(dt_filename)[0]
            pred_save_path = os.path.join(savedir, f"{dt_basename}_{count_suffix}{ap_suffix}.jpg")
            cv2.imwrite(pred_save_path, image)

        elif mask_on:
            # mask_on分支：根据开关加AP后缀
            for pred_r in pred_results:
                segmentation = pred_r['segmentation']
                mask = maskUtils.decode(segmentation)
                mask = mask.squeeze() if mask.ndim == 3 else mask
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = contours[0]
                    cat_id = int(pred_r['category_id'])
                    color = COLOR_MAP[cat_id]
                    cv2.drawContours(image, [cnt], 0, color, 3)
            # 构造保存路径
            base_name = '%012d' % image_id if is_coco_formatname else DTimages[:-4]
            final_base = f"{base_name}{ap_suffix}" if add_single_map_ap else base_name
            pred_save_path = os.path.join(savedir, f"{final_base}.jpg")
            cv2.imwrite(pred_save_path, image)
            # 生成GT mask图（原有逻辑）
            try:
                for gt_r in gt_results:
                    segmentation = gt_r['segmentation']
                    mask = maskUtils.decode(segmentation)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = contours[0]
                        cat_id = int(gt_r['category_id'])
                        color = COLOR_MAP.get(cat_id, (128, 128, 128))
                        cv2.drawContours(imagegt, [cnt], 0, color, 3)
                gt_dir = os.path.join(savedir[:savedir.rfind('/')], 'GT')
                os.makedirs(gt_dir, exist_ok=True)
                gt_base = f"%012d{ap_suffix}" % image_id if add_single_map_ap else '%012d' % image_id
                cv2.imwrite(os.path.join(gt_dir, f"{gt_base}.jpg"), imagegt)
            except:
                pass

        else:
            # 默认分支：绘制bbox，根据开关加AP后缀
            for ori_cat_id, ori_cat_key in enumerate(ORI_DATACATS):
                selected_class_set.append(ORI_DATACATS[ori_cat_key]['name'])
            for cat_id, cat_key in enumerate(cats):
                cat = cats[cat_key]
                if cat['name'] in selected_class_set:
                    this_image_pred_ious = cocoEval.ious.get((image_id, cat_key), [])
                    if not this_image_pred_ious:
                        continue
                    if cat_key not in gt_catkeys:
                        continue
                    for box_id in range(min(1000, this_image_pred_ious.shape[0])):
                        USE_THIS_BOX = False
                        box_ious = this_image_pred_ious[box_id, :]
                        possible_gtbox_id = np.argmax(box_ious) if box_ious.size > 0 else -1
                        if possible_gtbox_id >= 0:
                            ious_of_gt = this_image_pred_ious[:, possible_gtbox_id]
                            prefered_box_id = np.argmax(ious_of_gt) if ious_of_gt.size > 0 else -1
                        else:
                            prefered_box_id = -1
                        try:
                            this_box = cocoEval.cocoDt.anns[
                                cocoEval.evalImgs[cat_id * 4 * len(image_ids) + iter]['dtIds'][box_id]]
                        except:
                            continue
                        if mask_on or vis_every_box:
                            USE_THIS_BOX = True
                        if possible_gtbox_id >= 0 and np.max(box_ious) >= THRE:
                            try:
                                gt_id = cocoEval.evalImgs[cat_id * 4 * len(image_ids) + iter]['gtIds'][
                                    possible_gtbox_id]
                                if gt_id not in DETECTED_GTBOX_ID:
                                    color = (51, 204, 51)
                                    DETECTED_GTBOX_ID.append(gt_id)
                                    if count_green_ap:
                                        this_im_green_counter += 1
                                    if box_id == prefered_box_id:
                                        USE_THIS_BOX = True
                                else:
                                    USE_THIS_BOX = False
                            except:
                                USE_THIS_BOX = False
                        elif this_box['score'] > THRE:
                            USE_THIS_BOX = True
                            color = (0, 255, 255)
                            currentim_OBJ_count += 1
                        else:
                            USE_THIS_BOX = False
                        # 绘制bbox
                        if use_class_colormap:
                            cat_id = int(this_box['category_id'])
                            color = COLOR_MAP.get(cat_id, (128, 128, 128))
                        if USE_THIS_BOX:
                            x, y, w, h = this_box['bbox']
                            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                            FOUND_DETECT_OBJECT = True
                            # 绘制标签
                            text = f"{cat['name']}:{this_box['score'] * 100:.1f}%"
                            txt_color = (0, 0, 0) if np.mean(color) > 127.5 else (255, 255, 255)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            txt_size = cv2.getTextSize(text, font, 0.4 * (image.shape[1] / 500), 1)[0]
                            txt_bk = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
                            if NEED_LABELBOX:
                                cv2.rectangle(image, (int(x), int(y) + 1),
                                              (int(x) + txt_size[0] + 1, int(y) + int(1.5 * txt_size[1])), txt_bk, -1)
                                cv2.putText(image, text, (x, y + txt_size[1]), font, 0.4, txt_color, 1)
            # 构造保存路径
            base_name = '%012d' % image_id if is_coco_formatname else f"{DTimages[:-4]}_count{currentim_OBJ_count}"
            final_base = f"{base_name}{ap_suffix}" if add_single_map_ap else base_name
            pred_save_path = os.path.join(savedir, f"{final_base}.jpg")
            if FOUND_DETECT_OBJECT:
                cv2.imwrite(pred_save_path, image)

        # ========== 7. 拼接pred和GT可视化图（原有逻辑） ==========
        if pred_save_path and os.path.exists(pred_save_path):
            # 从cocoGT的file_name解析GT可视化图路径
            gt_file_name = cocoEval.cocoGt.imgs[image_id]['file_name']
            gt_vis_path = _get_gt_vis_path(gt_file_name, gt_vis_dir)

            if gt_vis_path:
                # 构造拼接后的保存路径（保存在comp文件夹，文件名与pred一致）
                comp_filename = os.path.basename(pred_save_path)
                comp_save_path = os.path.join(comp_dir, comp_filename)
                # 执行拼接
                if _merge_pred_gt_images(pred_save_path, gt_vis_path, comp_save_path):
                    comp_processed += 1
                    print(f"✅ 拼接完成：{comp_save_path}")
                else:
                    comp_skipped += 1
            else:
                print(f"❌ 未找到图片ID {image_id} 的GT可视化图，跳过拼接")
                comp_skipped += 1
        else:
            print(f"❌ 图片ID {image_id} 的pred可视化图未生成，跳过拼接")
            comp_skipped += 1

        # 计数统计（原有逻辑）
        if count_green_ap:
            print(f'image NO.{image_id} :   detected {len(pred_results)} ---- GT all {len(gt_results)}')
            TOTAL_green_counter += len(pred_results)
            TOTAL_GT_counter += len(gt_results)

        # ========== 新增：计算当前图片的病理指标 ==========
        if compute_pathology_metrics_flag:
            img_metrics = compute_pathology_metrics(
                gt_input=cocoEval.cocoGt,
                dt_input=cocoEval.cocoDt,
                image_ids=[image_id]
            )['per_image_metrics'][0]
            img_metrics['map'] = single_map  # 补充map值
            per_image_metrics_list.append(img_metrics)
            print(
                f"📊 图片ID {image_id} 指标：DICE={img_metrics['DICE']:.3f}, HD={img_metrics['HD']:.2f}, AJI={img_metrics['AJI']:.3f}, DICEobj={img_metrics['DICEobj']:.3f}, PQ={img_metrics['PQ']:.3f}, F1={img_metrics['F1']:.3f}, map={img_metrics['map']:.4f}")

    # ========== 8. 最终统计 ==========
    if count_green_ap:
        acc = TOTAL_green_counter / TOTAL_GT_counter if TOTAL_GT_counter > 0 else 0
        print(f'accuracy: {acc}, detected {TOTAL_green_counter} ---- GT all {TOTAL_GT_counter}')
    # 拼接统计
    print(f"\n================ 拼接完成 ================")
    print(f"成功拼接pred+GT图数量：{comp_processed}")
    print(f"拼接跳过数量：{comp_skipped}")
    print(f"拼接图保存路径：{comp_dir}")

    # ========== 新增：全局病理指标计算与保存 ==========
    if compute_pathology_metrics_flag and per_image_metrics_list:
        # 计算全局指标
        global_metrics = {
            'mean_DICE': round(np.nanmean([m['DICE'] for m in per_image_metrics_list]), 3),
            'mean_HD': round(np.nanmean([m['HD'] for m in per_image_metrics_list]), 2),
            'mean_AJI': round(np.nanmean([m['AJI'] for m in per_image_metrics_list]), 3),
            'mean_DICEobj': round(np.nanmean([m['DICEobj'] for m in per_image_metrics_list]), 3),
            'mean_PQ': round(np.nanmean([m['PQ'] for m in per_image_metrics_list]), 3),
            'mean_F1': round(np.nanmean([m['F1'] for m in per_image_metrics_list]), 3),
            'mean_map': round(np.nanmean([m['map'] for m in per_image_metrics_list]), 4),
            'total_images': len(per_image_metrics_list),
            'valid_images': len([m for m in per_image_metrics_list if not np.isnan(m['DICE'])])
        }

        # 打印全局指标
        print(f"\n================ 全局病理指标 ================")
        print(f"平均DICE：{global_metrics['mean_DICE']:.3f}")
        print(f"平均HD95：{global_metrics['mean_HD']:.2f}")
        print(f"平均AJI：{global_metrics['mean_AJI']:.3f}")
        print(f"平均DICEobj：{global_metrics['mean_DICEobj']:.3f}")
        print(f"平均PQ：{global_metrics['mean_PQ']:.3f}")
        print(f"平均F1-score：{global_metrics['mean_F1']:.3f}")
        print(f"平均mAP：{global_metrics['mean_map']:.4f}")
        print(f"有效图片数：{global_metrics['valid_images']}/{global_metrics['total_images']}")

        # 保存指标到CSV
        if metrics_save_path is None:
            metrics_save_path = os.path.join(savedir, "pathology_metrics.csv")
        # 保存单图指标
        metrics_df = pd.DataFrame(per_image_metrics_list)
        metrics_df = metrics_df[['image_id', 'DICE', 'HD', 'AJI', 'DICEobj', 'PQ', 'F1', 'map', 'TP', 'FP', 'FN']]
        metrics_df.to_csv(metrics_save_path, index=False, encoding='utf-8')
        print(f"\n📁 指标已保存到：{metrics_save_path}")

        # 保存全局指标
        global_metrics_df = pd.DataFrame([global_metrics])
        global_metrics_path = os.path.join(savedir, "global_pathology_metrics.csv")
        global_metrics_df.to_csv(global_metrics_path, index=False, encoding='utf-8')
        print(f"📁 全局指标已保存到：{global_metrics_path}")


# ==================== 新增：独立调用指标计算的示例 ====================
def calculate_metrics_from_json(gt_json_path, dt_json_path, save_dir):
    """
    从JSON文件计算病理指标（独立调用示例）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 计算指标
    metrics_result = compute_pathology_metrics(
        gt_input=gt_json_path,
        dt_input=dt_json_path
    )

    # 保存结果
    # 单图指标
    per_image_df = pd.DataFrame(metrics_result['per_image_metrics'])
    per_image_df.to_csv(os.path.join(save_dir, "per_image_metrics.csv"), index=False)

    # 全局指标
    global_df = pd.DataFrame([metrics_result['global_metrics']])
    global_df.to_csv(os.path.join(save_dir, "global_metrics.csv"), index=False)

    # 打印结果
    print("全局指标：")
    for k, v in metrics_result['global_metrics'].items():
        print(f"{k}: {v}")


# ==================== 测试调用示例 ====================
if __name__ == "__main__":
    # 示例1：独立计算JSON文件的指标
    # calculate_metrics_from_json(
    #     gt_json_path="/home/data/jy/GLIP/DATASET/coco2s/annotations/instances_val2017_4class.json",
    #     dt_json_path="你的预测结果JSON路径",
    #     save_dir="./metrics_results"
    # )

    # 示例2：调用可视化函数（含指标计算）
    # 需要先初始化cocoEval对象，示例：
    # coco_gt = COCO("/home/data/jy/GLIP/DATASET/coco2s/annotations/instances_val2017_4class.json")
    # coco_dt = coco_gt.loadRes("你的预测结果JSON路径")
    # coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    # coco_eval.params.imgIds = coco_gt.getImgIds()
    #
    # draw_2color_bboxes_on_images(
    #     cocoEval=coco_eval,
    #     savedir="./visual_results",
    #     compute_pathology_metrics_flag=True,
    #     metrics_save_path="./visual_results/pathology_metrics.csv"
    # )
    pass
def draw_3color_bboxes_on_images(cocoEval, savedir,valdata_dir='DATASETS/COCO/val2017',THRE=0.5):
    # 检查保存结果的文件夹是否存在，如果不存在则创建
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # 获取所有图片的ID
    image_ids = cocoEval.params.imgIds
    filename_key = 'file_name'
    for iter,image_id in enumerate(image_ids):
        # 获取当前图片的信息
        image_id=int(image_id)
        image_info = cocoEval.cocoGt.loadImgs(image_id)[0]
        try:
            image_path = os.path.join(image_info[filename_key])
        except:
            filename_key = 'filename'
            image_path = os.path.join(image_info[filename_key])

        image = cv2.imread(valdata_dir+'/'+image_path)
        if image is None:
            image = cv2.imread(valdata_dir + '/%012d.jpg'%image_id)
        gt_results = cocoEval.cocoGt.loadAnns(cocoEval.cocoGt.getAnnIds(imgIds=image_id))
        # 获取当前图片的预测结果
        pred_results = cocoEval.cocoDt.loadAnns(cocoEval.cocoDt.getAnnIds(imgIds=image_id))
        # pred_bboxes = [result for result in pred_results if result['image_id'] == image_id]

        # 绘制预测结果的bbox
        this_image_pred_ious=cocoEval.ious[(image_id,1)]
        DETECTED_GTBOX_ID=[]
        for box_id in range(min(400,this_image_pred_ious.shape[0])):
            USE_THIS_BOX=True
            box_ious=this_image_pred_ious[box_id,:]
            possible_gtbox_id=np.argmax(box_ious)
            try:
                this_box=cocoEval.cocoDt.anns[cocoEval.evalImgs[iter]['dtIds'][box_id]]
            except:
                print(box_id)
            if np.max(box_ious)>=0.5 :
                if possible_gtbox_id not in DETECTED_GTBOX_ID:
                    color = (51, 204, 51)  # 绿色
                    DETECTED_GTBOX_ID.append(possible_gtbox_id)
                else:
                    USE_THIS_BOX = False
            elif this_box['score']>THRE:
                color = (0, 0, 255)  # 红色
            else:
                USE_THIS_BOX=False
            if this_box['area']>60000:
                USE_THIS_BOX = False
            # 绘制bbox
            if USE_THIS_BOX:
                x, y, w, h = this_box['bbox']
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)
        for gtbox_id,gt_result in enumerate(gt_results):
            if gtbox_id not in DETECTED_GTBOX_ID:
                bbox = gt_result['bbox']
                color = (0, 255, 255)  # 黄色
                # 绘制bbox
                x, y, w, h = bbox
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        # 绘制未被预测到的GT实例的bbox
        # for gtbox_id,gt_result in enumerate(gt_results):
        #     gtbox_ious=this_image_pred_ious[:,gtbox_id]
        #     if np.max(gtbox_ious)<0.5:
        #         bbox = gt_result['bbox']
        #         color = (0, 255, 255)  # 黄色
        #         # 绘制bbox
        #         x, y, w, h = bbox
        #         cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 1)

        # 保存绘制结果的图片
        save_path = os.path.join(savedir, '%012d.jpg'%image_id)
        cv2.imwrite(save_path, image)
        
def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None,col=[255,255,255]):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2]+box[0])
        y1 = int(box[3]+box[1])

        # color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        # cv2.rectangle(
        #     img,
        #     (x0, y0 + 1),
        #     (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        #     txt_bk_color,
        #     -1
        # )
        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
def vis_old(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2]+box[0])
        y1 = int(box[3]+box[1])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
def vis_dataset(dataset,savdir='TOSHOW',TASK_DATASET='datasets/COCO/val2017'):#input json_format dataset
    import skimage.io as io
    for image_id in range(480):
        img = cv2.imread(TASK_DATASET+'/%012d.jpg' % (image_id))
        if img is None:
            continue
        # img = cv2.cvtColor(imori, cv2.COLOR_BGR2RGB)
        boxes=[ann['bbox'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        try:
            scores=[ann['score'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        except:
            scores=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        cls_ids=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        this_image_vis=vis(img, boxes, scores, cls_ids, conf=0.3, class_names=None)
        # plt.imshow(this_image_vis)
        # plt.show()
        cv2.imwrite('%s/%012d.jpg' % (savdir,image_id),this_image_vis)
def vis_multi_dataset(dataset,dataset2,savdir='TOSHOW'):#input json_format dataset
    import skimage.io as io
    for image_id in range(480):
        imori = cv2.imread('datasets/COCO/val2017/%012d.jpg' % (image_id))
        if imori is None:
            continue
        img = cv2.cvtColor(imori, cv2.COLOR_BGR2RGB)
        boxes=[ann['bbox'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        try:
            scores=[ann['score'] for ann in dataset['annotations'] if ann['image_id']==image_id]
        except:
            scores=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        cls_ids=[1 for ann in dataset['annotations'] if ann['image_id']==image_id]
        this_image_vis=vis(img, boxes, scores, cls_ids, conf=0.2, class_names=None)
        boxes = [ann['bbox'] for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        try:
            scores = [ann['score'] for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        except:
            scores = [1 for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        cls_ids = [1 for ann in dataset2['annotations'] if ann['image_id'] == image_id]
        this_image_vis = vis(img, boxes, scores, cls_ids, conf=0.1, class_names=None,col=[0,255,0])
        # plt.imshow(this_image_vis)
        # plt.show()
        cv2.imwrite('%s/%012d.jpg' % (savdir,image_id),this_image_vis)

_COLORS = np.array(
    [
        0.000, 0, 1,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
import numpy as np
import cv2
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from scipy.optimize import linear_sum_assignment

import numpy as np
from scipy.optimize import linear_sum_assignment

def get_fast_pq_from_masks(true_masks, pred_masks, match_iou=0.5):
    """
    使用匈牙利算法从 binary mask 列表计算 Panoptic Quality（PQ）。
    参数:
        true_masks: List[np.ndarray]，每个是 shape=(H, W) 的 bool mask。
        pred_masks: List[np.ndarray]，每个是 shape=(H, W) 的 bool mask。
        match_iou: float，匹配所需的最小 IoU 阈值（默认 0.5）。
    返回:
        dq: Detection Quality
        sq: Segmentation Quality
        pq: Panoptic Quality
    """
    assert match_iou >= 0.0, "IoU threshold must be non-negative"

    if len(true_masks) == 0 and len(pred_masks) == 0:
        return 1.0, 1.0, 1.0
    if len(true_masks) == 0:
        return 0.0, 0.0, 0.0
    if len(pred_masks) == 0:
        return 0.0, 0.0, 0.0

    num_true = len(true_masks)
    num_pred = len(pred_masks)
    iou_matrix = np.zeros((num_true, num_pred), dtype=np.float64)

    # 计算 pairwise IoU
    for t_idx, t_mask in enumerate(true_masks):
        for p_idx, p_mask in enumerate(pred_masks):
            inter = np.logical_and(t_mask, p_mask).sum()
            if inter > 0:
                union = np.logical_or(t_mask, p_mask).sum()
                iou_matrix[t_idx, p_idx] = inter / (union + 1e-6)

    # 一对一匈牙利匹配，目标是最大化 IoU（等价于最小化 -IoU）
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= match_iou:
            matched_pairs.append((r, c))

    if len(matched_pairs) == 0:
        return 0.0, 0.0, 0.0

    paired_true, paired_pred = zip(*matched_pairs)
    paired_true = np.array(paired_true)
    paired_pred = np.array(paired_pred)
    paired_iou = iou_matrix[paired_true, paired_pred]

    # 计算 TP / FP / FN
    tp = len(paired_true)
    fp = num_pred - tp
    fn = num_true - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
    sq = paired_iou.sum() / (tp + 1e-6)
    pq = dq * sq
    

    return paired_iou.sum(),tp, fp, fn


def get_fast_aji_from_masks(true_masks, pred_masks, height, width):
    """
    直接从 binary mask 列表计算 AJI，不依赖 instance map，不受重叠问题影响
    true_masks: List[np.ndarray]，每个元素是 shape=(H, W) 的 bool mask
    pred_masks: List[np.ndarray]，同上
    """
    # fast check
    if len(pred_masks) == 0:
        return 0.0

    # 创建 true_id/pred_id 和 mask 的映射
    true_id_list = list(range(1, len(true_masks) + 1))
    pred_id_list = list(range(1, len(pred_masks) + 1))
    true_masks_map = [None] + true_masks
    pred_masks_map = [None] + pred_masks

    # 初始化
    pairwise_inter = np.zeros((len(true_masks), len(pred_masks)), dtype=np.float64)
    pairwise_union = np.zeros((len(true_masks), len(pred_masks)), dtype=np.float64)

    for t_idx, t_mask in enumerate(true_masks, start=1):
        # 找到 t_mask 中重叠的预测实例
        for p_idx, p_mask in enumerate(pred_masks, start=1):
            inter = np.logical_and(t_mask, p_mask).sum()
            if inter > 0:
                union = np.logical_or(t_mask, p_mask).sum()
                pairwise_inter[t_idx - 1, p_idx - 1] = inter
                pairwise_union[t_idx - 1, p_idx - 1] = union

    pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)
    matched_pred = np.argmax(pairwise_iou, axis=1)
    matched_iou = np.max(pairwise_iou, axis=1)

    paired_true = np.nonzero(matched_iou > 0.0)[0]
    paired_pred = matched_pred[paired_true]

    overall_inter = pairwise_inter[paired_true, paired_pred].sum()
    overall_union = pairwise_union[paired_true, paired_pred].sum()

    # 统计未匹配的 GT 和 Pred
    matched_true_ids = set(paired_true.tolist())
    matched_pred_ids = set(paired_pred.tolist())

    for t_idx in range(len(true_masks)):
        if t_idx not in matched_true_ids:
            overall_union += true_masks[t_idx].sum()
    for p_idx in range(len(pred_masks)):
        if p_idx not in matched_pred_ids:
            overall_union += pred_masks[p_idx].sum()

    return overall_inter , overall_union

def compute_metrics(gt_annotation_path, pred_annotation_path):
    # 初始化 COCO 对象
    coco_gt = COCO(gt_annotation_path)
    coco_pred = COCO(pred_annotation_path)
    img_ids = coco_gt.getImgIds()

    # 全局统计量
    sum_iou_total = 0.0
    tp_total = 0
    fp_total = 0
    fn_total = 0
    total_dice = 0.0
    total_diceobj = 0.0
    total_hd = 0.0
    total_aji = 0.0
    count_pairs = 0
    num_images = len(img_ids)

    for img_id in img_ids:
        # 加载图像尺寸信息
        img_info = coco_gt.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']

        # 加载标注数据
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

        # 处理 GT 的掩膜
        gt_masks = []

        for ann in gt_anns:
            segm = ann['segmentation']
            if isinstance(segm, list):
                rle = maskUtils.frPyObjects(segm, height, width)
                mask = maskUtils.decode(rle)
                mask = mask.any(axis=2) if mask.ndim == 3 else mask
                gt_masks.append(mask.astype(bool))
            else:
                mask = maskUtils.decode(segm)
                mask = mask.squeeze() if mask.ndim == 3 else mask
                gt_masks.append(mask.astype(bool))

        # 处理 Pred 的掩膜
        pred_masks = []
        for ann in pred_anns:
            segm = ann['segmentation']
            if isinstance(segm, list):
                rle = maskUtils.frPyObjects(segm, height, width)
                mask = maskUtils.decode(rle)
                mask = mask.any(axis=2) if mask.ndim == 3 else mask
                pred_masks.append(mask.astype(bool))
            else:
                mask = maskUtils.decode(segm)
                mask = mask.squeeze() if mask.ndim == 3 else mask
                pred_masks.append(mask.astype(bool))

        num_gt = len(gt_masks)
        num_pred = len(pred_masks)
        iou_matrix = np.zeros((num_gt, num_pred))

        for i in range(num_gt):
            for j in range(num_pred):
                intersection = np.logical_and(gt_masks[i], pred_masks[j]).sum()
                union = np.logical_or(gt_masks[i], pred_masks[j]).sum()
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0

        # 匹配
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_pairs = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= 0.5]

        # 统计匹配情况
        tp = len(matched_pairs)
        fp = num_pred - tp
        fn = num_gt - tp

        sum_iou_total += sum(iou_matrix[r, c] for r, c in matched_pairs)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        # Diceobj、HD、AJI
        aji_numerator = 0
        matched_pred_indices = []
        for r, c in matched_pairs:
            gt_mask = gt_masks[r]
            pred_mask = pred_masks[c]
            matched_pred_indices.append(c)

            # Diceobj
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = gt_mask.sum() + pred_mask.sum()
            total_diceobj += 2 * intersection / union if union > 0 else 0

            # Hausdorff
            gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            gt_points = np.vstack([cnt[:, 0, :] for cnt in gt_contours]) if gt_contours else np.zeros((0, 2))
            pred_points = np.vstack([cnt[:, 0, :] for cnt in pred_contours]) if pred_contours else np.zeros((0, 2))
            if gt_points.size > 0 and pred_points.size > 0:
                hd = max(directed_hausdorff(gt_points, pred_points)[0],
                         directed_hausdorff(pred_points, gt_points)[0])
            else:
                hd = max(height, width)
            total_hd += hd

            # AJI numerator
            aji_numerator += intersection

        # AJI denominator
        unmatched_pred_indices = [c for c in range(num_pred) if c not in matched_pred_indices]
        aji_denominator = sum(mask.sum() for mask in gt_masks) + \
                          sum(pred_masks[c].sum() for c in unmatched_pred_indices)
        total_aji += aji_numerator / aji_denominator if aji_denominator > 0 else 0

        count_pairs += tp

        # 图像级 Dice
        if num_gt == 0 and num_pred == 0:
            total_dice += 1.0
        else:
            gt_union = np.logical_or.reduce(gt_masks, axis=0) if num_gt > 0 else np.zeros((height, width), bool)
            pred_union = np.logical_or.reduce(pred_masks, axis=0) if num_pred > 0 else np.zeros((height, width), bool)
            intersection = np.logical_and(gt_union, pred_union).sum()
            union = gt_union.sum() + pred_union.sum()
            total_dice += 2 * intersection / union if union > 0 else 0.0

    avg_dice = total_dice / num_images if num_images > 0 else 0.0
    avg_diceobj = total_diceobj / count_pairs if count_pairs > 0 else 0.0
    avg_hd = total_hd / count_pairs if count_pairs > 0 else 0.0
    avg_aji = total_aji / num_images if num_images > 0 else 0.0

    if tp_total + fp_total + fn_total == 0:
        pq = 1.0
    else:
        if tp_total == 0:
            sq = 0.0
            rq = 0.0
        else:
            sq = sum_iou_total / tp_total
            rq = tp_total / (tp_total + 0.5 * (fp_total + fn_total))
        pq = sq * rq

    metrics = {
        'Dice': avg_dice,
        'Diceobj': avg_diceobj,
        'HD': avg_hd,
        'AJI': avg_aji,
        'PQ': pq,
    }
    return metrics

def compute_metrics2(gt_annotation_path, pred_annotation_path):
    if isinstance(gt_annotation_path, COCO):
        coco_gt = gt_annotation_path
        coco_pred = pred_annotation_path
    else:
        coco_gt = COCO(gt_annotation_path)
        # coco_pred = COCO(pred_annotation_path)
        with open(gt_annotation_path, 'r') as f:
            gt_data = json.load(f)

        # Load your prediction annotations
        with open(pred_annotation_path, 'r') as f:
            pred_anns = json.load(f)

        # Create a new COCO dataset structure using the GT structure but replacing annotations
        pred_data = {
            'info': gt_data.get('info', {}),
            'licenses': gt_data.get('licenses', []),
            'categories': gt_data.get('categories', []),
            'images': gt_data.get('images', []),
            'annotations': pred_anns  # This replaces the ground truth annotations
        }
        # Save to a temporary file (COCO needs a file path to initialize)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(pred_data, f)
            f.flush()
            # Initialize COCO with the prediction data
            coco_pred = COCO(f.name)
    img_ids = coco_gt.getImgIds()
    category_ids = coco_gt.getCatIds()
    TOTAL_cat_wise_anns_count=0
    avg_aji = 0
    avg_diceobj = 0
    avg_dice = 0
    avg_hd = 0
    avg_pq=0
    for cat_id in category_ids:
        # 全局统计量
        sum_iou_total = 0.0
        tp_total = 0
        fp_total = 0
        fn_total = 0
        total_dice = 0.0
        total_diceobj = 0.0
        total_hd = 0.0
        total_aji = 0.0
        total_pq = 0.0
        total_sq = 0.0
        total_dq = 0.0
        count_pairs = 0
        total_paired_iou =0
        TOTAL_cat_wise_anns_count = 0
        total_fp =0
        total_tp = 0
        total_fn = 0
        num_images = len(img_ids)
        total_overall_inter = 0
        total_overall_union = 0
        cat_wise_anns_count=0
        for img_id in img_ids:
            # 加载图像尺寸信息
            img_info = coco_gt.loadImgs(img_id)[0]
            height, width = img_info['height'], img_info['width']

            # 加载标注数据
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
            pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

            # 处理 GT 的多边形分割掩膜
            gt_masks = []
            for ann in gt_anns:
                segm = ann['segmentation']
                THIS_category_id=ann['category_id']
                if THIS_category_id==cat_id:
                    cat_wise_anns_count+=1
                    TOTAL_cat_wise_anns_count+=1
                else:
                    continue
                # 处理多边形格式（单实例可能包含多个多边形）
                if isinstance(segm, list):
                    rle = maskUtils.frPyObjects(segm, height, width)
                    mask = maskUtils.decode(rle)
                    mask = mask.any(axis=2) if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))
                else:
                    mask = maskUtils.decode(segm)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))
            if cat_wise_anns_count>0:
                pred_masks = []
                for ann in pred_anns:
                    segm = ann['segmentation']
                    score = ann['score']
                    THIS_category_id = ann['category_id']
                    if THIS_category_id!=cat_id:
                        continue
                    if score < 0.3:
                        continue
                    if isinstance(segm, list):
                        rle = maskUtils.frPyObjects(segm, height, width)
                        mask = maskUtils.decode(rle)
                        mask = mask.any(axis=2) if mask.ndim == 3 else mask
                        pred_masks.append(mask.astype(bool))
                    else:
                        mask = maskUtils.decode(segm)
                        mask = mask.squeeze() if mask.ndim == 3 else mask
                        pred_masks.append(mask.astype(bool))
                        # raise ValueError("Pred segmentation 必须是多边形列表")
                # 计算 IoU 矩阵（匈牙利算法匹配用）
                num_gt = len(gt_masks)
                num_pred = len(pred_masks)
                iou_matrix = np.zeros((num_gt, num_pred))

                for i in range(num_gt):
                    gt_mask = gt_masks[i]
                    for j in range(num_pred):
                        pred_mask = pred_masks[j]
                        intersection = np.logical_and(gt_mask, pred_mask).sum()
                        union = np.logical_or(gt_mask, pred_mask).sum()
                        iou_matrix[i, j] = intersection / union if union > 0 else 0.0

                # 匈牙利算法匹配（最大化 IoU 总和）
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_pairs = []
                for r, c in zip(row_ind, col_ind):
                    if iou_matrix[r, c] >= 0.5:  # IoU 阈值设为 0.5
                        matched_pairs.append((r, c))

                # 统计 TP/FP/FN
                tp = len(matched_pairs)
                fp = num_pred - tp
                fn = num_gt - tp

                # 更新 PQ 的全局统计量
                sum_iou = sum(iou_matrix[r, c] for r, c in matched_pairs)
                sum_iou_total += sum_iou
                tp_total += tp
                fp_total += fp
                fn_total += fn

                # 计算 Diceobj 和 HD（仅匹配对）
                for r, c in matched_pairs:
                    gt_mask = gt_masks[r]
                    pred_mask = pred_masks[c]

                    # Dice 系数（对象级别）
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum())
                    total_diceobj += dice

                    # Hausdorff 距离（需要轮廓点）
                    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # 提取轮廓点
                    gt_points = np.vstack([cnt[:,0,:] for cnt in gt_contours]) if gt_contours else np.zeros((0,2))
                    pred_points = np.vstack([cnt[:,0,:] for cnt in pred_contours]) if pred_contours else np.zeros((0,2))

                    # 计算双向 Hausdorff 距离
                    if gt_points.size > 0 and pred_points.size > 0:
                        hd1 = directed_hausdorff(gt_points, pred_points)[0]
                        hd2 = directed_hausdorff(pred_points, gt_points)[0]
                        hd = max(hd1, hd2)
                    else:
                        hd = max(height, width)  # 无轮廓时用图像对角线长度作为默认值
                    total_hd += hd
                    count_pairs += 1

                # 计算图像级 Dice（合并所有掩膜）
                if num_gt == 0 and num_pred == 0:
                    dice_image = 1.0  # 双方均为空掩膜
                else:
                    gt_combined = np.logical_or.reduce(gt_masks, axis=0) if num_gt > 0 else np.zeros((height, width), bool)
                    pred_combined = np.logical_or.reduce(pred_masks, axis=0) if num_pred > 0 else np.zeros((height, width), bool)
                    intersection = np.logical_and(gt_combined, pred_combined).sum()
                    dice_image = 2 * intersection / (gt_combined.sum() + pred_combined.sum()) if (gt_combined.sum() + pred_combined.sum()) > 0 else 0.0
                total_dice += dice_image * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) #in case of TOAL=0
                try:
                    overall_inter , overall_union = get_fast_aji_from_masks(gt_masks, pred_masks, height, width)
                except:
                    overall_inter, overall_union =(1,1)
                # paired_iou ,tp, fp, fn  = get_fast_pq_from_masks(gt_masks, pred_masks)
                # total_paired_iou += paired_iou
                # total_tp += tp
                # total_fp += fp
                # total_fn += fn
                total_overall_inter +=overall_inter
                total_overall_union +=overall_union
        # avg_dq = total_tp / (total_tp + 0.5 * total_fp + 0.5 * total_fn + 1e-6)
        # avg_sq = total_paired_iou / (total_tp + 1e-6)
        # avg_pq = avg_dq * avg_sq
        avg_aji += (1/len(category_ids))* total_overall_inter/total_overall_union * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6))
        # 计算最终指标
        # avg_dq = total_dq /num_images if num_images > 0 else 0.0
        # avg_sq = total_sq /num_images if num_images > 0 else 0.0
        # avg_pq = total_pq /num_images if num_images > 0 else 0.0
        # avg_aji = total_aji /num_images if num_images > 0 else 0.0
        avg_dice += (1/len(category_ids)) * (total_dice / num_images) * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if num_images > 0 else 0.0
        avg_diceobj += (1/len(category_ids)) * (total_diceobj / count_pairs) * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if count_pairs > 0 else 0.0
        avg_hd += (1/len(category_ids)) * (total_hd / count_pairs) * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if count_pairs > 0 else 0.0

        # 计算 PQ（全景质量）
        if tp_total + fp_total + fn_total == 0:
            pq = 1.0  # 所有实例均正确匹配且无 FP/FN
        else:
            if tp_total == 0:
                sq = 0.0
                rq = 0.0
            else:
                sq = sum_iou_total / tp_total  # 平均匹配质量
                rq = tp_total / (tp_total + 0.5 * (fp_total + fn_total))  # 识别质量
            pq = sq * rq
        avg_pq += (1/len(category_ids)) * pq * (cat_wise_anns_count / (TOTAL_cat_wise_anns_count + 1e-6)) if count_pairs > 0 else 0.0

    metrics= {
        'Dice': avg_dice,
        'HD': avg_hd,
        'PQ': avg_pq,
        'Diceobj': avg_diceobj,
        'AJI': avg_aji
    }
    # print(f"pq_new: {avg_pq:.4f}")
    # print(f"dq_new: {avg_dq:.4f}")
    # print(f"sq_new: {avg_sq:.4f}")
    
    # print(f"Hausdorff Distance: {metrics['HD']:.2f} pixels")
    # print(f"Panoptic Quality (PQ): {metrics['PQ']:.4f}")
    # print(f"Dice (Object Level): {metrics['Diceobj']:.4f}")
    return metrics


from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import hausdorff_distance as directed_hausdorff
import cv2
import pycocotools.mask as maskUtils


def compute_metrics2new(gt_annotation_path, pred_annotation_path):
    if isinstance(gt_annotation_path, COCO):
        coco_gt = gt_annotation_path
        coco_pred = pred_annotation_path
    else:
        coco_gt = COCO(gt_annotation_path)
        # coco_pred = COCO(pred_annotation_path)
        with open(gt_annotation_path, 'r') as f:
            gt_data = json.load(f)

        # Load your prediction annotations
        with open(pred_annotation_path, 'r') as f:
            pred_anns = json.load(f)

        # Create a new COCO dataset structure using the GT structure but replacing annotations
        pred_data = {
            'info': gt_data.get('info', {}),
            'licenses': gt_data.get('licenses', []),
            'categories': gt_data.get('categories', []),
            'images': gt_data.get('images', []),
            'annotations': pred_anns  # This replaces the ground truth annotations
        }

        # Save to a temporary file (COCO needs a file path to initialize)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            json.dump(pred_data, f)
            f.flush()
            # Initialize COCO with the prediction data
            coco_pred = COCO(f.name)
    img_ids = coco_gt.getImgIds()
    category_ids = coco_gt.getCatIds()

    # 初始化按类别存储的统计量
    class_metrics = defaultdict(lambda: {
        'sum_iou_total': 0.0,
        'tp_total': 0,
        'fp_total': 0,
        'fn_total': 0,
        'total_dice': 0.0,
        'total_diceobj': 0.0,
        'total_hd': 0.0,
        'total_overall_inter': 0,
        'total_overall_union': 0,
        'count_pairs': 0,
        'num_images': 0
    })

    # 全局统计量（所有类别）
    global_metrics = {
        'sum_iou_total': 0.0,
        'tp_total': 0,
        'fp_total': 0,
        'fn_total': 0,
        'total_dice': 0.0,
        'total_diceobj': 0.0,
        'total_hd': 0.0,
        'total_overall_inter': 0,
        'total_overall_union': 0,
        'count_pairs': 0,
        'num_images': len(img_ids)
    }

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']

        # 按类别组织标注
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

        # 按类别分组
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)

        for ann in gt_anns:
            gt_by_class[ann['category_id']].append(ann)

        for ann in pred_anns:
            pred_by_class[ann['category_id']].append(ann)

        # 处理每个类别的标注
        for cat_id in set(gt_by_class.keys()).union(pred_by_class.keys()):
            class_metrics[cat_id]['num_images'] += 1

            # 获取当前类别的标注
            cat_gt_anns = gt_by_class.get(cat_id, [])
            cat_pred_anns = pred_by_class.get(cat_id, [])

            # 处理GT和预测的掩膜
            gt_masks = []
            for ann in cat_gt_anns:
                segm = ann['segmentation']
                if isinstance(segm, list):
                    rle = maskUtils.frPyObjects(segm, height, width)
                    mask = maskUtils.decode(rle)
                    mask = mask.any(axis=2) if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))
                else:
                    mask = maskUtils.decode(segm)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    gt_masks.append(mask.astype(bool))

            pred_masks = []
            for ann in cat_pred_anns:
                segm = ann['segmentation']
                if isinstance(segm, list):
                    rle = maskUtils.frPyObjects(segm, height, width)
                    mask = maskUtils.decode(rle)
                    mask = mask.any(axis=2) if mask.ndim == 3 else mask
                    pred_masks.append(mask.astype(bool))
                else:
                    mask = maskUtils.decode(segm)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    pred_masks.append(mask.astype(bool))

            # 计算IoU矩阵
            num_gt = len(gt_masks)
            num_pred = len(pred_masks)
            iou_matrix = np.zeros((num_gt, num_pred))

            for i in range(num_gt):
                gt_mask = gt_masks[i]
                for j in range(num_pred):
                    pred_mask = pred_masks[j]
                    intersection = np.logical_and(gt_mask, pred_mask).sum()
                    union = np.logical_or(gt_mask, pred_mask).sum()
                    iou_matrix[i, j] = intersection / union if union > 0 else 0.0

            # 匈牙利算法匹配
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_pairs = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= 0.5:  # IoU阈值
                    matched_pairs.append((r, c))

            # 统计TP/FP/FN
            tp = len(matched_pairs)
            fp = num_pred - tp
            fn = num_gt - tp

            # 更新类别统计量
            sum_iou = sum(iou_matrix[r, c] for r, c in matched_pairs)
            class_metrics[cat_id]['sum_iou_total'] += sum_iou
            class_metrics[cat_id]['tp_total'] += tp
            class_metrics[cat_id]['fp_total'] += fp
            class_metrics[cat_id]['fn_total'] += fn

            # 计算匹配对的指标
            for r, c in matched_pairs:
                gt_mask = gt_masks[r]
                pred_mask = pred_masks[c]

                # Dice系数
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum())
                class_metrics[cat_id]['total_diceobj'] += dice

                # Hausdorff距离
                gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)

                gt_points = np.vstack([cnt[:, 0, :] for cnt in gt_contours]) if gt_contours else np.zeros((0, 2))
                pred_points = np.vstack([cnt[:, 0, :] for cnt in pred_contours]) if pred_contours else np.zeros((0, 2))

                if gt_points.size > 0 and pred_points.size > 0:
                    hd1 = directed_hausdorff(gt_points, pred_points)[0]
                    hd2 = directed_hausdorff(pred_points, gt_points)[0]
                    hd = max(hd1, hd2)
                else:
                    hd = max(height, width)
                class_metrics[cat_id]['total_hd'] += hd
                class_metrics[cat_id]['count_pairs'] += 1

            # 计算图像级Dice
            if num_gt == 0 and num_pred == 0:
                dice_image = 1.0
            else:
                gt_combined = np.logical_or.reduce(gt_masks, axis=0) if num_gt > 0 else np.zeros((height, width), bool)
                pred_combined = np.logical_or.reduce(pred_masks, axis=0) if num_pred > 0 else np.zeros((height, width),
                                                                                                       bool)
                intersection = np.logical_and(gt_combined, pred_combined).sum()
                dice_image = 2 * intersection / (gt_combined.sum() + pred_combined.sum()) if (
                                                                                                         gt_combined.sum() + pred_combined.sum()) > 0 else 0.0
            class_metrics[cat_id]['total_dice'] += dice_image

            # 计算AJI相关统计量
            overall_inter, overall_union = get_fast_aji_from_masks(gt_masks, pred_masks, height, width)
            class_metrics[cat_id]['total_overall_inter'] += overall_inter
            class_metrics[cat_id]['total_overall_union'] += overall_union

            # 更新全局统计量
            global_metrics['sum_iou_total'] += sum_iou
            global_metrics['tp_total'] += tp
            global_metrics['fp_total'] += fp
            global_metrics['fn_total'] += fn
            global_metrics['total_diceobj'] += class_metrics[cat_id]['total_diceobj']
            global_metrics['total_hd'] += class_metrics[cat_id]['total_hd']
            global_metrics['count_pairs'] += class_metrics[cat_id]['count_pairs']
            global_metrics['total_overall_inter'] += overall_inter
            global_metrics['total_overall_union'] += overall_union

    # 计算每个类别的指标
    results = {}
    for cat_id in category_ids:
        metrics = class_metrics[cat_id]
        num_images = metrics['num_images']
        count_pairs = metrics['count_pairs']

        # Dice
        avg_dice = metrics['total_dice'] / num_images if num_images > 0 else 0.0

        # HD
        avg_hd = metrics['total_hd'] / count_pairs if count_pairs > 0 else 0.0

        # PQ
        if metrics['tp_total'] + metrics['fp_total'] + metrics['fn_total'] == 0:
            pq = 1.0
        else:
            if metrics['tp_total'] == 0:
                sq = 0.0
                rq = 0.0
            else:
                sq = metrics['sum_iou_total'] / metrics['tp_total']
                rq = metrics['tp_total'] / (metrics['tp_total'] + 0.5 * (metrics['fp_total'] + metrics['fn_total']))
            pq = sq * rq

        # Diceobj
        avg_diceobj = metrics['total_diceobj'] / count_pairs if count_pairs > 0 else 0.0

        # AJI
        avg_aji = metrics['total_overall_inter'] / metrics['total_overall_union'] if metrics[
                                                                                         'total_overall_union'] > 0 else 0.0

        results[f'class_{cat_id}'] = {
            'Dice': avg_dice,
            'HD': avg_hd,
            'PQ': pq,
            'Diceobj': avg_diceobj,
            'AJI': avg_aji,
            'Num_Images': num_images
        }

    # 计算全局指标（所有类别）
    num_images = global_metrics['num_images']
    count_pairs = global_metrics['count_pairs']

    # Dice
    avg_dice = global_metrics['total_dice'] / num_images if num_images > 0 else 0.0

    # HD
    avg_hd = global_metrics['total_hd'] / count_pairs if count_pairs > 0 else 0.0

    # PQ
    if global_metrics['tp_total'] + global_metrics['fp_total'] + global_metrics['fn_total'] == 0:
        pq = 1.0
    else:
        if global_metrics['tp_total'] == 0:
            sq = 0.0
            rq = 0.0
        else:
            sq = global_metrics['sum_iou_total'] / global_metrics['tp_total']
            rq = global_metrics['tp_total'] / (
                        global_metrics['tp_total'] + 0.5 * (global_metrics['fp_total'] + global_metrics['fn_total']))
        pq = sq * rq

    # Diceobj
    avg_diceobj = global_metrics['total_diceobj'] / count_pairs if count_pairs > 0 else 0.0

    # AJI
    avg_aji = global_metrics['total_overall_inter'] / global_metrics['total_overall_union'] if global_metrics[
                                                                                                   'total_overall_union'] > 0 else 0.0

    results['overall'] = {
        'Dice': avg_dice,
        'HD': avg_hd,
        'PQ': pq,
        'Diceobj': avg_diceobj,
        'AJI': avg_aji,
        'Num_Images': num_images
    }

    return results['overall']

# 辅助函数（需要实现）
import json
import copy
import numpy as np
from typing import Dict, List, Tuple
from pycocotools.coco import COCO
import numpy as np
import cv2
import copy
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
def compute_all_metrics(gt_annotation_path, pred_annotation_path):
    # ==================== 基础指标计算 ====================
    # 初始化COCO对象
    coco_gt = COCO(gt_annotation_path)
    coco_pred = COCO(pred_annotation_path)
    img_ids = coco_gt.getImgIds()

    # 初始化统计量
    metrics = {
        # 分割指标
        'Dice': 0.0, 'HD': 0.0, 'AJI': 0.0, 'Diceobj': 0.0,
        # 检测指标
        'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'AR': 0.0
    }

    # ==================== 分割指标计算 ====================
    sum_iou_total = 0.0
    total_aji = 0.0
    count_pairs = 0
    num_images = len(img_ids)

    for img_id in img_ids:
        # 加载图像信息
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']

        # 加载标注并生成掩膜
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        pred_anns = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=img_id))

        gt_masks = [_poly_to_mask(ann['segmentation'], h, w) for ann in gt_anns]
        pred_masks = [_poly_to_mask(ann['segmentation'], h, w) for ann in pred_anns]

        # 计算IoU矩阵
        iou_matrix = _compute_iou_matrix(gt_masks, pred_masks)

        # 匈牙利匹配
        matched_pairs, tp, fp, fn = _hungarian_matching(iou_matrix)

        # 更新统计量
        sum_iou_total += sum(iou_matrix[r, c] for r, c in matched_pairs)
        count_pairs += len(matched_pairs)

        # 计算实例级指标
        aji_numerator = 0.0
        for r, c in matched_pairs:
            # Diceobj
            intersection = (gt_masks[r] & pred_masks[c]).sum()
            metrics['Diceobj'] += 2 * intersection / (gt_masks[r].sum() + pred_masks[c].sum())

            # HD
            hd = _calculate_hd(gt_masks[r], pred_masks[c], h, w)
            metrics['HD'] += hd

            # AJI分子
            aji_numerator += (gt_masks[r] & pred_masks[c]).sum()

        # AJI分母
        matched_pred_indices = [c for _, c in matched_pairs]
        unmatched_pred_indices = [c for c in range(len(pred_masks)) if c not in matched_pred_indices]

        aji_denominator = (
                sum(gt_mask.sum() for gt_mask in gt_masks) +
                sum(pred_masks[c].sum() for c in unmatched_pred_indices)
        )
        total_aji += aji_numerator / aji_denominator if aji_denominator > 0 else 0

        # 图像级Dice
        gt_combined = np.logical_or.reduce(gt_masks) if gt_masks else np.zeros((h, w), bool)
        pred_combined = np.logical_or.reduce(pred_masks) if pred_masks else np.zeros((h, w), bool)
        intersection = (gt_combined & pred_combined).sum()
        metrics['Dice'] += 2 * intersection / (gt_combined.sum() + pred_combined.sum()) if (
                                                                                                       gt_combined.sum() + pred_combined.sum()) > 0 else 0

    # 归一化指标
    metrics['Dice'] /= num_images
    metrics['Diceobj'] /= count_pairs if count_pairs > 0 else 1
    metrics['HD'] /= count_pairs if count_pairs > 0 else 1
    metrics['AJI'] = total_aji / num_images

    # # ==================== 检测指标计算 ====================
    # # 运行COCO官方评估
    # coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    #
    # # 提取指标
    # metrics['mAP'] = coco_eval.stats[0]
    # metrics['AP50'] = coco_eval.stats[1]
    # metrics['AP75'] = coco_eval.stats[2]
    # metrics['AR'] = coco_eval.stats[8]

    return metrics


# ==================== 工具函数 ====================
def _poly_to_mask(segmentation, height, width):
    """将多边形转换为二值掩膜"""
    rle = maskUtils.frPyObjects(segmentation, height, width)
    mask = maskUtils.decode(rle)
    return mask.any(axis=2) if mask.ndim == 3 else mask


def _compute_iou_matrix(gt_masks, pred_masks):
    """计算IoU矩阵"""
    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
    for i, gm in enumerate(gt_masks):
        for j, pm in enumerate(pred_masks):
            intersection = (gm & pm).sum()
            union = (gm | pm).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    return iou_matrix


def _hungarian_matching(iou_matrix, iou_threshold=0.5):
    """匈牙利算法匹配"""
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= iou_threshold]
    tp = len(matched)
    fp = iou_matrix.shape[1] - tp
    fn = iou_matrix.shape[0] - tp
    return matched, tp, fp, fn


def _calculate_hd(gt_mask, pred_mask, img_h, img_w):
    """计算Hausdorff距离"""
    # 提取轮廓 (修复参数传递错误)
    gt_contours, _ = cv2.findContours(
        gt_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    pred_contours, _ = cv2.findContours(
        pred_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # 生成轮廓点集合 (修复列表推导式括号错误)
    gt_points = np.vstack([c[:, 0, :] for c in gt_contours]) if gt_contours else np.zeros((0, 2))
    pred_points = np.vstack([c[:, 0, :] for c in pred_contours]) if pred_contours else np.zeros((0, 2))

    # 处理空掩膜情况
    if gt_points.size == 0 or pred_points.size == 0:
        return np.sqrt(img_h ** 2 + img_w ** 2)  # 返回图像对角线长度

    # 计算双向Hausdorff距离 (修复max函数调用)
    hd_forward = directed_hausdorff(gt_points, pred_points)[0]
    hd_backward = directed_hausdorff(pred_points, gt_points)[0]
    return max(hd_forward, hd_backward)

# gt_ann_path = "/home/data/jy/GLIP/DATASET/coco1/annotations/instances_train2017.json"
# pred_ann_path = "/home/data/jy/GLIP/maskrcnn_benchmark/utils/dice07.json"
# # 使用示例
# metrics = compute_all_metrics(
#     gt_annotation_path=gt_ann_path,
#     pred_annotation_path=pred_ann_path
# )
#
# print(f"""
# 分割指标:
# Dice: {metrics['Dice']:.4f}
# Diceobj: {metrics['Diceobj']:.4f}
# AJI: {metrics['AJI']:.4f}
# HD: {metrics['HD']:.2f} px
#
# 检测指标:
# mAP: {metrics['mAP']:.4f}
# AP50: {metrics['AP50']:.4f}
# AP75: {metrics['AP75']:.4f}
# AR: {metrics['AR']:.4f}
# """)

if __name__ == '__main__':
    # # 使用示例
    gt_ann_path = "/home/data/jy/GLIP/DATASET/coco1s/annotations/instances_train2017.json"
    pred_ann_path = "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.22305/mask.json"#"/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.28100/mask.json"# "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.22411/mask.json"
    # # perturbed_data = generate_perturbed_gt(
    # #     gt_json_path=gt_ann_path,
    # #     output_json_path='dice07.json',
    # #     target_dice=0.7
    # # )
    #
    metrics = compute_metrics2(gt_ann_path, pred_ann_path)
    print(f"Dice (Image Level): {metrics['Dice']:.4f}")
    print(f"Hausdorff Distance: {metrics['HD']:.2f} pixels")
    print(f"Panoptic Quality (PQ): {metrics['PQ']:.4f}")
    print(f"Dice (Object Level): {metrics['Diceobj']:.4f}")
    print(f"AJI: {metrics['AJI']:.4f}")
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    将掩码以半透明颜色覆盖到原图上，非掩码区域保持原图不变
    Args:
        image: 原图 (H, W, 3), uint8 格式
        mask: 二值掩码 (H, W), bool类型或0/1
        color: 覆盖颜色，默认绿色 (R, G, B)
        alpha: 透明度 (0为完全透明，1为不透明)
    Returns:
        叠加后的图像 (H, W, 3), uint8
    """
    # 确保输入合法性
    assert image.ndim == 3 and image.shape[2] == 3, "原图必须是HWC格式的三通道图像"
    mask = mask.astype(bool)  # 强制转换为布尔掩码
    if mask.shape != image.shape[:2]:
        raise ValueError("掩码与原图的尺寸不匹配")

    # 转换为浮点类型以便计算
    image_float = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    color_layer = image_float.copy()  # 初始化颜色层为原图

    # 只在掩码区域应用颜色覆盖
    color_layer[mask] = np.array(color) / 255.0  # 将颜色归一化到 [0, 1]

    # 混合原图和颜色层（仅在掩码区域混合）
    blended = image_float.copy()
    blended[mask] = (image_float[mask] * (1 - alpha) +
                     color_layer[mask] * alpha)

    # 转换回uint8格式
    blended = (blended * 255).clip(0, 255).astype(np.uint8)
    return blended
tumors=[2,14,15,16,27,28,29,30,40,41,42,43,44,53,54,55,56,57,58,66,67,68,69,70,71,105,106,107,108,109,118,119,120,121,122,131,132,133,134,135,144,145,146,147,157,158,159,160,170,171,
        261,262,263,264,274,275,276,277,287,220,231,232,233,244,245,246,248,259,271,272,283,284,285,299,317,318,329,330,332,342,343,344,345,356,355,357,358,371,362,363,364,375,376,377,388,389,402,
        436,437,438,449,450,451,452,453,454,464,465,466,476,477,478,489,490,491,492,503,504]
import copy
import io
import sys
from pycocotools.cocoeval import COCOeval

import sys
import io
import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_segmentation_metrics(mask_gt, mask_pred):
    """
    计算单张图片的分割指标（DICE、HD95、AJI、DICEobj、PQ、F1-score）
    注：HD95需要安装scikit-image，AJI/PQ参考医学图像分割标准实现

    参数:
        mask_gt: 标注掩码 (numpy array, shape [H,W], dtype uint8)
        mask_pred: 预测掩码 (numpy array, shape [H,W], dtype uint8)

    返回:
        dict: 包含所有分割指标的字典
    """
    metrics = {
        'DICE': 0.0,
        'HD95': 0.0,
        'AJI': 0.0,
        'DICEobj': 0.0,
        'PQ': 0.0,
        'F1-score': 0.0
    }

    # 空掩码处理
    if np.sum(mask_gt) == 0 and np.sum(mask_pred) == 0:
        metrics['DICE'] = 1.0
        metrics['F1-score'] = 1.0
        metrics['AJI'] = 1.0
        metrics['DICEobj'] = 1.0
        metrics['PQ'] = 1.0
        return metrics
    if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
        return metrics

    try:
        # 1. DICE系数 ( Sørensen-Dice Coefficient )
        intersection = np.sum(mask_gt * mask_pred)
        union = np.sum(mask_gt) + np.sum(mask_pred)
        metrics['DICE'] = 2.0 * intersection / union if union > 0 else 0.0

        # 2. F1-score (与DICE等价，这里单独计算保持兼容性)
        tp = intersection
        fp = np.sum(mask_pred) - tp
        fn = np.sum(mask_gt) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['F1-score'] = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # 3. HD95 (Hausdorff Distance 95th percentile)
        try:
            from skimage.metrics import hausdorff_distance
            # 获取轮廓点计算HD95
            gt_coords = np.argwhere(mask_gt > 0)
            pred_coords = np.argwhere(mask_pred > 0)
            if len(gt_coords) > 0 and len(pred_coords) > 0:
                hd = hausdorff_distance(gt_coords, pred_coords)
                metrics['HD95'] = hd
            else:
                metrics['HD95'] = -1.0
        except ImportError:
            metrics['HD95'] = -1.0  # 标记未安装scikit-image
        except Exception as e:
            metrics['HD95'] = -2.0  # 标记计算错误

        # 4. AJI (Aggregated Jaccard Index) - 医学图像分割常用指标
        intersection_aji = np.sum(mask_gt * mask_pred)
        union_aji = np.sum(np.logical_or(mask_gt, mask_pred))
        metrics['AJI'] = intersection_aji / union_aji if union_aji > 0 else 0.0

        # 5. DICEobj (目标级DICE，与像素级区分)
        metrics['DICEobj'] = metrics['DICE']  # 基础实现，可根据需求细化

        # 6. PQ (Panoptic Quality) - 简化版实现
        # PQ = (IOU * TP) / (TP + 0.5*FP + 0.5*FN)
        iou = intersection / union if union > 0 else 0.0
        metrics['PQ'] = (iou * tp) / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0

    except Exception as e:
        print(f"⚠️ 分割指标计算失败: {str(e)}")

    return metrics


def get_mask_from_annotations(coco_obj, img_id, cat_id=None):
    """
    从COCO标注中获取指定图片的掩码

    参数:
        coco_obj: COCO对象（gt或dt）
        img_id: 图片ID
        cat_id: 类别ID（None表示所有类别）

    返回:
        numpy array: 掩码数组 (H,W)
    """
    img_info = coco_obj.imgs.get(img_id)
    if not img_info:
        return None

    # 创建空掩码
    mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

    # 获取标注
    ann_ids = coco_obj.getAnnIds(imgIds=[img_id], catIds=[cat_id] if cat_id else [])
    anns = coco_obj.loadAnns(ann_ids)

    # 绘制掩码
    for ann in anns:
        if 'segmentation' in ann and ann['segmentation']:
            # 处理COCO格式的segmentation
            if isinstance(ann['segmentation'], list):
                # 多边形格式
                from pycocotools import mask as mask_utils
                if len(ann['segmentation']) > 0 and isinstance(ann['segmentation'][0], list):
                    # 转换为RLE
                    rle = mask_utils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                    ann_mask = mask_utils.decode(rle)
                    mask = np.maximum(mask, ann_mask)
            elif isinstance(ann['segmentation'], dict):
                # RLE格式
                from pycocotools import mask as mask_utils
                ann_mask = mask_utils.decode(ann['segmentation'])
                mask = np.maximum(mask, ann_mask)

    return mask


def list_single_map_ap(coco_eval_obj):
    """
    从COCOeval对象中提取并简洁打印每张图片的详细评估指标
    包含：mAP、AP50、AP75、AR，以及分割指标（DICE、HD95、AJI、DICEobj、PQ、F1-score）
    并计算所有指标的数据集平均值

    参数:
        coco_eval_obj: 已初始化的COCOeval对象（需关联coco_gt和coco_pred）

    返回:
        tuple: (all_metrics, avg_metrics)
            - all_metrics: dict {img_id: {指标名: 数值}} 所有图片的详细指标
            - avg_metrics: dict {指标名: 平均值} 数据集级别的平均指标
    """
    # 存储所有图片的指标结果
    all_metrics = {}
    avg_metrics = {}

    # 1. 基础校验
    if not hasattr(coco_eval_obj, 'cocoGt') or not hasattr(coco_eval_obj, 'cocoDt'):
        print("错误：COCOeval对象未关联cocoGt/cocoDt！")
        return all_metrics, avg_metrics

    coco_gt = coco_eval_obj.cocoGt
    coco_pred = coco_eval_obj.cocoDt
    iou_type = coco_eval_obj.params.iouType

    # 2. 获取图片ID列表
    img_ids = coco_eval_obj.params.imgIds if coco_eval_obj.params.imgIds else list(coco_gt.imgs.keys())
    if not img_ids:
        print("警告：无待评估的图片ID！")
        return all_metrics, avg_metrics

    # 3. 打印表头
    print(f"\n===== 单图详细评估指标列表（{iou_type}）=====")
    print("=" * 140)
    header = (
        f"{'图片名称':60s} | {'mAP':8s} | {'AP50':8s} | {'AP75':8s} | "
        f"{'AR':8s} | {'DICE':8s} | {'HD95':8s} | {'AJI':8s} | "
        f"{'DICEobj':8s} | {'PQ':8s} | {'F1-score':8s}"
    )
    print(header)
    print("-" * 140)
    valid_count = 0

    # 4. 遍历计算单图指标
    for img_id in img_ids:
        img_metrics = {'img_id': img_id, 'file_name': ''}

        # 获取图片信息
        img_info = coco_gt.imgs.get(img_id)
        img_name = img_info['file_name'] if img_info else f"未知图片_ID-{img_id}"
        img_metrics['file_name'] = img_name

        # ========== 1. 计算检测指标（mAP/AP50/AP75/AR） ==========
        # 深拷贝参数，避免污染原对象
        single_eval = COCOeval(coco_gt, coco_pred, iou_type)
        single_eval.params = copy.deepcopy(coco_eval_obj.params)
        single_eval.params.imgIds = [img_id]

        # 屏蔽summarize()的默认打印输出
        sys.stdout = io.StringIO()
        try:
            single_eval.evaluate()
            single_eval.accumulate()
            single_eval.summarize()
        except Exception as e:
            sys.stdout = sys.__stdout__
            print(
                f"{img_name:60s} | {'计算失败':8s} | {'-':8s} | {'-':8s} | {'-':8s} | {'-':8s} | {'-':8s} | {'-':8s} | {'-':8s} | {'-':8s} | {'-':8s}")
            print(f"图片ID-{img_id}：检测指标计算失败 - {str(e)}")
            continue
        finally:
            sys.stdout = sys.__stdout__

        # 提取检测指标
        if hasattr(single_eval, 'stats') and len(single_eval.stats) >= 12:
            # stats说明：
            # 0: AP @[0.5:0.95] (mAP)
            # 1: AP @0.5 (AP50)
            # 2: AP @0.75 (AP75)
            # 8: AR @[0.5:0.95] (maxDets=100)
            img_metrics['mAP'] = single_eval.stats[0]
            img_metrics['AP50'] = single_eval.stats[1]
            img_metrics['AP75'] = single_eval.stats[2]
            img_metrics['AR'] = single_eval.stats[8]

            # 过滤mAP=0的情况（可选）
            if img_metrics['mAP'] < 1e-6:
                continue
        else:
            continue

        # ========== 2. 计算分割指标（DICE/HD95/AJI等） ==========
        try:
            # 获取GT和预测掩码
            gt_mask = get_mask_from_annotations(coco_gt, img_id)
            pred_mask = get_mask_from_annotations(coco_pred, img_id)

            if gt_mask is not None and pred_mask is not None:
                # 计算分割指标
                seg_metrics = calculate_segmentation_metrics(gt_mask, pred_mask)
                img_metrics.update(seg_metrics)
            else:
                # 掩码获取失败
                img_metrics.update({
                    'DICE': -1.0, 'HD95': -1.0, 'AJI': -1.0,
                    'DICEobj': -1.0, 'PQ': -1.0, 'F1-score': -1.0
                })
        except Exception as e:
            print(f"⚠️ 图片ID-{img_id}分割指标计算失败: {str(e)}")
            img_metrics.update({
                'DICE': -2.0, 'HD95': -2.0, 'AJI': -2.0,
                'DICEobj': -2.0, 'PQ': -2.0, 'F1-score': -2.0
            })

        # ========== 3. 打印结果 ==========
        # 格式化输出（处理特殊值）
        def format_val(val):
            if val < 0:
                return "N/A"
            # HD95保留1位小数，其他保留4位
            return f"{val:.1f}" if 'HD95' in str(val) else f"{val:.4f}"

        print(
            f"{img_name:60s} | "
            f"{format_val(img_metrics['mAP']):8s} | "
            f"{format_val(img_metrics['AP50']):8s} | "
            f"{format_val(img_metrics['AP75']):8s} | "
            f"{format_val(img_metrics['AR']):8s} | "
            f"{format_val(img_metrics.get('DICE', 0)):8s} | "
            f"{format_val(img_metrics.get('HD95', 0)):8s} | "
            f"{format_val(img_metrics.get('AJI', 0)):8s} | "
            f"{format_val(img_metrics.get('DICEobj', 0)):8s} | "
            f"{format_val(img_metrics.get('PQ', 0)):8s} | "
            f"{format_val(img_metrics.get('F1-score', 0)):8s}"
        )

        # 保存结果
        all_metrics[img_id] = img_metrics
        valid_count += 1

    # ========== 4. 计算完整的数据集平均指标 ==========
    print("-" * 140)
    print(f"有效图片数（mAP>0）：{valid_count} / 总图片数：{len(img_ids)}")

    # 定义所有需要计算平均值的指标
    all_metrics_list = [
        'mAP', 'AP50', 'AP75', 'AR',
        'DICE', 'HD95', 'AJI', 'DICEobj', 'PQ', 'F1-score'
    ]

    # 初始化平均值字典
    avg_metrics = {}
    # 同时统计有效样本数
    valid_samples_count = {}

    # 计算每个指标的平均值
    for metric in all_metrics_list:
        # 收集所有有效值（排除-1/-2等错误标记）
        valid_values = []
        for img_metrics in all_metrics.values():
            val = img_metrics.get(metric, -1)
            if val >= 0:  # 只计算有效值
                valid_values.append(val)

        # 计算平均值
        if valid_values:
            avg_metrics[metric] = np.mean(valid_values)
            valid_samples_count[metric] = len(valid_values)
        else:
            avg_metrics[metric] = 0.0
            valid_samples_count[metric] = 0

    # ========== 5. 打印格式化的平均值汇总 ==========
    print("\n" + "=" * 80)
    print("📊 数据集级别平均指标汇总")
    print("=" * 80)

    # 分两组打印：检测指标 和 分割指标
    print("\n【检测指标】")
    detection_metrics = ['mAP', 'AP50', 'AP75', 'AR']
    for metric in detection_metrics:
        if avg_metrics[metric] > 0:
            print(f"  • 平均{metric}: {avg_metrics[metric]:.4f} (有效样本数: {valid_samples_count[metric]})")
        else:
            print(f"  • 平均{metric}: N/A (有效样本数: {valid_samples_count[metric]})")

    print("\n【分割指标】")
    segmentation_metrics = ['DICE', 'HD95', 'AJI', 'DICEobj', 'PQ', 'F1-score']
    for metric in segmentation_metrics:
        if avg_metrics[metric] > 0:
            if metric == 'HD95':
                print(f"  • 平均{metric}: {avg_metrics[metric]:.1f} (有效样本数: {valid_samples_count[metric]})")
            else:
                print(f"  • 平均{metric}: {avg_metrics[metric]:.4f} (有效样本数: {valid_samples_count[metric]})")
        else:
            print(f"  • 平均{metric}: N/A (有效样本数: {valid_samples_count[metric]})")

    print("\n" + "=" * 80)

    # ========== 6. 返回结果 ==========
    return all_metrics, avg_metrics


# 示例使用代码
if __name__ == "__main__":
    # 示例：如何使用这个函数
    """
    # 1. 加载GT和预测
    coco_gt = COCO('path/to/gt.json')
    coco_dt = coco_gt.loadRes('path/to/pred.json')

    # 2. 初始化COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')  # 'segm' for segmentation, 'bbox' for bounding box
    coco_eval.params.imgIds = coco_gt.getImgIds()  # 评估所有图片
    coco_eval.params.catIds = coco_gt.getCatIds()  # 评估所有类别

    # 3. 计算并打印单图指标和平均值
    all_metrics, avg_metrics = list_single_map_ap(coco_eval)

    # 4. 使用结果
    print(f"\n📈 所有图片的详细指标已保存，共{len(all_metrics)}张图片")
    print(f"📊 数据集平均mAP: {avg_metrics.get('mAP', 0):.4f}")
    print(f"📊 数据集平均DICE: {avg_metrics.get('DICE', 0):.4f}")
    """
    pass