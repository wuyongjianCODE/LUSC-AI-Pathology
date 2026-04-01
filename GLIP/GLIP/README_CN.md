# 基于GLIP的肺肿瘤分析

本仓库为 GLIP (Grounding Language-Image Pre-training) 的修改版本，面向肺肿瘤分析与WSI处理。

## 1. 环境要求

- CUDA 11.0+
- Python 3.7+
- 16GB+ RAM
- 8GB+ 显存

依赖安装:
1. 按 GLIP 官方文档安装基础依赖
2. 安装本仓库依赖:
```bash
pip install -e .
```

## 2. 数据集

数据位于 `DATASET/`，包含:
- SVS WSI 文件
- 预处理 patch
- 标注文件

## 3. 训练命令 (26672)

以下为 示范命令
```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python tools/train_net.py --config-file configs/pretrain/LUNG2.yaml --restart False MODEL.RETINANET.DETECTIONS_PER_IMG 400 MODEL.FCOS.DETECTIONS_PER_IMG 400 MODEL.ATSS.DETECTIONS_PER_IMG 400 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 400 THRE 0.5 TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 SOLVER.CHECKPOINT_PERIOD 000 SOLVER.MAX_ITER 4000 DATASETS.TEST "('lungtest4_2025_val',)" DATASETS.TRAIN "('lungtest3_grounding_train',)" MODEL.WEIGHT "/home/data/jy/GLIP/OUTPUT_TRAIN_lungscc2BKUP/model_final.pth"
```

## 4. 权重使用

1. 下载 `model_final.pth` (论文随附权重)
2. 放到任意目录
3. 修改命令中的 `MODEL.WEIGHT` 路径，例如:
```bash
MODEL.WEIGHT "/absolute/path/to/model_final.pth"
```

## 5. 输出

运行后会在 GLIP 工作目录下生成 `OUTPUT/`，包含可视化结果。

## 6. 预训练权重

最终权重位置:
```
F:\wyj\NATURE\GLIP\model_final.pth
```

## 7. 关键功能 (script3.py)

`script3.py` 实现智能WSI patch提取:
1. 区域检测: 基于缩略图颜色分割
2. Patch 选择: 优先靠近肿瘤区域
3. COCO 标注生成
4. 可选细胞检测与统计

用法示例:
```bash
python script3.py --pred_dir /path/to/thumbnails --wsi_dir /path/to/wsi/files --k 4 --thre 10000
```

参数说明:
- `--pred_dir`: 缩略图目录
- `--wsi_dir`: WSI 目录
- `--k`: 扩展范围
- `--thre`: 每张 WSI 的最大 patch 数

## 8. 目录结构
```
GLIP/
  configs/             # 配置文件
  DATASET/             # 数据集
  tools/               # 训练与评估脚本
  maskrcnn_benchmark/  # 修改后的 maskrcnn
  script3.py           # WSI patch 提取
  setup.py             # 安装脚本
  model_final.pth      # 预训练权重
```

## 9. 注意事项

- WSI 文件需为 `.svs`
- 大规模 WSI 处理需要较大内存
- 可按 GPU 显存调整 batch size
