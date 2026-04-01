# 项目总览

本目录包含论文中两部分代码:

1. 分类/肿瘤负荷评估 (RVT/MPR)  
   见 `F:\wyj\NATURE\lung\lung\README_CN.md` 与 `F:\wyj\NATURE\lung\lung\README_ENG.md`

2. 检测/细胞与区域检测 (GLIP)  
   见 `F:\wyj\NATURE\GLIP\GLIP\README_CN.md` 与 `F:\wyj\NATURE\GLIP\GLIP\README_ENG.md`

论文与代码对应关系 (简要):
- RVT/MPR 分类评估: `lung/` 目录 (`predict.py` 与 `LungNPresnet_C5_BIGDATA_MULTISCALE_shenzhen.py`)
- 细胞与区域检测: `GLIP/` 目录

## 大文件与下载说明

由于 GitHub 仓库不适合直接存放大体积权重、全量数据集和 WSI 样本图，本仓库当前仅公开代码、配置文件和必要的说明文档。以下内容未直接上传到 GitHub 仓库：

- 训练权重与模型文件
  - `GLIP/GLIP/model_final.pth`（约 5.1 GB）
  - `lung/lung/weights/model_final.h5`
  - 其他中间 checkpoint / `.pth` / `.h5` 文件
- 数据集与中间产物
  - `GLIP/GLIP/DATASET/`
  - `lung/lung/samples/`
  - 由脚本生成的可视化、patch、统计结果

推荐发布方式：

- 小于 2 GB 的单个模型权重：优先放到 GitHub Releases
- 大于 2 GB 的模型权重、完整数据集、WSI 样本：优先放到 Hugging Face / Zenodo / 机构网盘
- 若数据涉及版权、患者隐私或第三方使用协议：只提供申请方式、预处理脚本和元数据，不直接公开原始文件

建议链接位（后续可直接替换）：

- 模型权重下载：
  - `GLIP main checkpoint`: `TBD`
  - `lung classification weights`: `TBD`
- 数据集下载：
  - `Processed lung patches / metadata`: `TBD`
  - `Example WSI samples (if permitted)`: `TBD`
- 发布页：
  - `GitHub Releases`: `https://github.com/wuyongjianCODE/LUSC-AI-Pathology/releases`
  - `Project repository`: `https://github.com/wuyongjianCODE/LUSC-AI-Pathology`

若后续提供外链，建议在本节补充：

1. 文件名称
2. 文件大小
3. SHA256 校验值
4. 下载链接
5. 使用限制 / 引用要求
