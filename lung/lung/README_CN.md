# 肺WSI分类复现指南

本目录对应论文中的“分类/肿瘤负荷评估（RVT/MPR）”部分。根据服务器 CUDA 版本使用不同入口脚本。

## 1. 环境与依赖

### CUDA 12 及以上
- 入口脚本: `predict.py`
- 依赖: 按 `requirements.txt` 安装

### CUDA 11 及以下
- 入口脚本: `LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py`
- 依赖: 同上，但老服务器可能需要自行适配更旧的 CUDA/TF 版本

安装:
```bash
pip install -r requirements.txt
```

## 2. 模型权重与示例数据

为便于直接复现，请在本目录下准备:
- `weights/model_final.h5`  (最终权重文件)
- `samples/*.svs`            (任意 TCGA 官方 SVS 文件)

当前已提供示例:
- `samples/20-02253-132023-06-30_09_34_19.svs`

说明:
- `predict.py` 的默认权重与输出路径写在脚本里，你可以在命令行覆盖。
- 如果要使用本仓库自带权重/样例，请把路径替换成你本机路径。

## 3. CUDA 12+ (predict.py)

`predict.py` 内部已写了范例用法，推荐从以下最小示例开始。

快速体验(直接运行):
```bash
bash demo.sh
```

单张 WSI:
```bash
python predict.py --wsi /abs/path/to/slide.svs \
  --weights /abs/path/to/model_final.h5 \
  --output /abs/path/to/output_predict/
```

批量 WSI:
```bash
python predict.py --wsi_dir /abs/path/to/svs_dir \
  --weights /abs/path/to/model_final.h5 \
  --output /abs/path/to/output_predict/
```

主要参数 (来自 `predict.py`):
- `--wsi` / `--wsi_dir`: 单张或批量 WSI 输入
- `--weights`: 权重文件路径
- `--output`: 输出目录
- `--batch_size`: 预测 batch size (默认 1024)
- `--patch_size`: patch 尺寸 (默认 224)
- `--thumb_height`: 缩略图高度 (默认 2000)
- `--fg_method`: 前景分割方法 (`ensemble|saturation|lab|deconv|texture`)
- `--extract_patches`: 开启 patch 提取
- `--patch_output_dir`: patch 输出目录

输出内容:
- `annotation/` 与 `overlay/` 缩略图
- 若启用 patch 提取: `patches/` 与 `visualizations/`

验证运行:
demo.sh  或
```bash
python predict.py \
  --wsi /data4/深圳分院35例/1号盒/20-02253_svs/20-02253-132023-06-30_09_34_19.svs \
  --weights /home/deeplearning/data/data2/wyj/test/weight/LUNG_NEW_final.h5 \
  --output /home/deeplearning/data/data2/wyj/test/output_predict_demo \
  --batch_size 32
```

输出示例:
- `output_predict_demo/annotation/20-02253-132023-06-30_09_34_19_mpr0.0034.png`
- `output_predict_demo/overlay/20-02253-132023-06-30_09_34_19_mpr0.0034.png`

如出现 `libcusolver.so.10` 缺失提示，说明 TF 版本与 CUDA 运行库不匹配，可能会退回 CPU 运行。
可通过安装 CUDA 兼容库或调整环境解决。

## 4. CUDA 11- (LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py)

该脚本注释较少，但可按以下方式运行。

训练:
```bash
python LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py train \
  --weights /abs/path/to/init_or_pretrained.h5 \
  --parallel 0
```

训练数据路径在脚本内部写死，需手动修改:
- `NP` 训练集根目录
- `NP_remained` 验证集根目录

测试/推理:
```bash
python LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py detect --start 0
```

需要手动修改的关键路径/参数:
- `SVSDIR`: WSI (SVS) 输入目录
- `itern`: 读取的权重编号 (默认 `LUNGBIGDATA_6.h5`)
- 输出目录: `TOSHOW_MULTISCALE_{itern}` (脚本自动生成)

运行后输出:
输出目录为 `TOSHOW_MULTISCALE_{itern}`，包含按整张 WSI 生成的可视化图:
`{svs}_whole_classify_mpr{mpr}.png`

## 5. 数据要求

输入可为任意 TCGA 官方 `.svs` 文件。请确保:
- WSI 文件可被 `openslide` 正常打开
- 服务器有足够内存 (建议 32GB+)

## 6. 常见问题

- **WSI 打不开**: 检查 `openslide` 安装与文件权限
- **显存不足**: 降低 `--batch_size`
- **无结果输出**: 确认 WSI 路径与权重路径是否正确
