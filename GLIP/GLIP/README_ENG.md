# GLIP-Based Lung Tumor Analysis

This repository is a modified GLIP (Grounding Language-Image Pre-training) project for lung tumor analysis and WSI processing.

## 1. Requirements

- CUDA 11.0+
- Python 3.7+
- 16GB+ RAM
- 8GB+ VRAM

Install dependencies:
1. Follow official GLIP installation
2. Install repo dependencies:
```bash
pip install -e .
```

## 2. Dataset

Located in `DATASET/`, including:
- SVS WSI files
- Preprocessed patches
- Annotation files

## 3. Training Command

Exact command :
```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python tools/train_net.py --config-file configs/pretrain/LUNG2.yaml --restart False MODEL.RETINANET.DETECTIONS_PER_IMG 400 MODEL.FCOS.DETECTIONS_PER_IMG 400 MODEL.ATSS.DETECTIONS_PER_IMG 400 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 400 THRE 0.5 TEST.IMS_PER_BATCH 1 SOLVER.IMS_PER_BATCH 1 SOLVER.CHECKPOINT_PERIOD 000 SOLVER.MAX_ITER 4000 DATASETS.TEST "('lungtest4_2025_val',)" DATASETS.TRAIN "('lungtest3_grounding_train',)" MODEL.WEIGHT "/home/data/jy/GLIP/OUTPUT_TRAIN_lungscc2BKUP/model_final.pth"
```

## 4. Released Weights

1. Download `model_final.pth`
2. Place it anywhere
3. Update `MODEL.WEIGHT`, e.g.:
```bash
MODEL.WEIGHT "/absolute/path/to/model_final.pth"
```

## 5. Output

An `OUTPUT/` folder will be created under the GLIP working directory with visualizations.

## 6. Pretrained Weights

Final weights location:
```
F:\wyj\NATURE\GLIP\model_final.pth
```

## 7. Key Feature (script3.py)

`script3.py` provides intelligent WSI patch extraction:
1. Region detection from thumbnails
2. Patch selection near tumor regions
3. COCO annotation generation
4. Optional cell detection/statistics

Example:
```bash
python script3.py --pred_dir /path/to/thumbnails --wsi_dir /path/to/wsi/files --k 4 --thre 10000
```

## 8. Directory Layout
```
GLIP/
  configs/
  DATASET/
  tools/
  maskrcnn_benchmark/
  script3.py
  setup.py
  model_final.pth
```

## 9. Notes

- WSI files should be `.svs`
- Large WSI processing requires substantial memory
- Tune batch size based on GPU VRAM
