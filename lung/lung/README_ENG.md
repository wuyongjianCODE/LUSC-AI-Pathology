# Lung WSI Classification Reproduction Guide

This directory corresponds to the paper’s classification/tumor burden evaluation (RVT/MPR) part. Choose the entry script based on your CUDA version.

## 1. Environment and Dependencies

### CUDA 12 and newer
- Entry script: `predict.py`
- Install dependencies from `requirements.txt`

### CUDA 11 and older
- Entry script: `LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py`
- Same dependencies, but older servers may need older CUDA/TF compatibility

Install:
```bash
pip install -r requirements.txt
```

## 2. Weights and Sample Data

Prepare in this directory:
- `weights/model_final.h5`  (final weights)
- `samples/*.svs`            (any official TCGA SVS file)

Provided sample:
- `samples/20-02253-132023-06-30_09_34_19.svs`

Notes:
- Default weight/output paths are defined in `predict.py`; you can override them via CLI.
- If using the bundled sample/weights, replace paths with your local paths.

## 3. CUDA 12+ (predict.py)

`predict.py` already includes usage examples. For a quick demo, just run:
```bash
bash demo.sh
```

Single WSI:
```bash
python predict.py --wsi /abs/path/to/slide.svs \
  --weights /abs/path/to/model_final.h5 \
  --output /abs/path/to/output_predict/
```

Batch WSI:
```bash
python predict.py --wsi_dir /abs/path/to/svs_dir \
  --weights /abs/path/to/model_final.h5 \
  --output /abs/path/to/output_predict/
```

Key parameters (from `predict.py`):
- `--wsi` / `--wsi_dir`: single or batch WSI input
- `--weights`: weights path
- `--output`: output directory
- `--batch_size`: prediction batch size (default 1024)
- `--patch_size`: patch size (default 224)
- `--thumb_height`: thumbnail height (default 2000)
- `--fg_method`: foreground segmentation (`ensemble|saturation|lab|deconv|texture`)
- `--extract_patches`: enable patch extraction
- `--patch_output_dir`: patch output directory

Outputs:
- `annotation/` and `overlay/` thumbnails
- If patch extraction enabled: `patches/` and `visualizations/`

Run Command:
demo.sh 
or
```bash
python predict.py \
  --wsi 20-02253-132023-06-30_09_34_19.svs \
  --weights /home/deeplearning/data/data2/wyj/test/weight/LUNG_NEW_final.h5 \
  --output /home/deeplearning/data/data2/wyj/test/output_predict_demo \
  --batch_size 32
```

Example outputs:
- `output_predict_demo/annotation/20-02253-132023-06-30_09_34_19_mpr0.0034.png`
- `output_predict_demo/overlay/20-02253-132023-06-30_09_34_19_mpr0.0034.png`

If `libcusolver.so.10` is missing, TF and CUDA runtime are mismatched and it may fall back to CPU.
Install CUDA compatibility libs or adjust the environment.

## 4. CUDA 11- (LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py)

Train:
```bash
python LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py train \
  --weights /abs/path/to/init_or_pretrained.h5 \
  --parallel 0
```

Hardcoded paths in the script must be updated:
- `NP` training dataset root
- `NP_remained` validation dataset root

Detect/infer:
```bash
python LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py detect --start 0
```

Key paths/params to edit:
- `SVSDIR`: WSI (SVS) input directory
- `itern`: weight index (default `LUNGBIGDATA_6.h5`)
- Output: `TOSHOW_MULTISCALE_{itern}` (auto-created)

## 5. Data Requirements

Input can be any official TCGA `.svs` file. Ensure:
- WSI can be opened by `openslide`
- Enough RAM (recommended 32GB+)

## 6. Troubleshooting

- **WSI cannot open**: check `openslide` install and file permissions
- **OOM**: reduce `--batch_size`
- **No outputs**: verify WSI path and weights path
