# -- coding: utf-8 --**
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""
import os
import json
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
# cocogt_dataset = json.load(f)
# newims = []
# x = os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
# x.sort()
# for image_id, name in enumerate(x):
#     newims.append({"id": int(name[:-4]),
#                    "height": 250, "width": 250, "file_name": name})
# cocogt_dataset['images'] = newims
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
# json.dump(cocogt_dataset, f)
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import torch
from skimage import io
torch.multiprocessing.set_sharing_strategy('file_system')
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import numpy as np
import random
from maskrcnn_benchmark.utils.amp import autocast, GradScaler
from pycocotools import mask as maskUtils
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def train(cfg, local_rank, distributed, use_tensorboard=False,):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if 'running_mean' in name:
                torch.nn.init.constant_(param, 0)
            if 'running_var' in name:
                torch.nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None

    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        print("LANGUAGE_BACKBONE FROZEN.")
        for p in model.language_backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False
    
    # if cfg.SOLVER.PROMPT_PROBING_LEVEL != -1:
    #     if cfg.SOLVER.PROMPT_PROBING_LEVEL == 1:
    #         for p in model.parameters():
    #             p.requires_grad = False

    #         for p in model.language_backbone.body.parameters():
    #             p.requires_grad = True

    #         for name, p in model.named_parameters():
    #             if p.requires_grad:
    #                 print(name, " : Not Frozen")
    #             else:
    #                 print(name, " : Frozen")
    #     else:
    #         assert(0)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=cfg.OUTPUT_DIR,
            start_iter=arguments["iteration"],
            delimiter="  "
        )
    else:
        meters = MetricLogger(delimiter="  ")
    if cfg.USE_TRAIN_COPY:
        from maskrcnn_benchmark.engine.trainer_copy import do_train
    else:
        from maskrcnn_benchmark.engine.trainer import do_train
    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        data_loaders_val,
        meters
    )

    return model

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def preprocess_raw_glip_result(jsonfile='LAST_PREDICT_BBOXS.json',visual=False):#wuyongjian edited : convert bboxs.json to a pseudo label jsonfile,which will be feed into the next cycle to fine-tune a new glip
    f=open(jsonfile,'r')
    cocodt_dataset_ann=json.load(f)
    f=open("DATASET/coco/annotations/instances_train2017_glipGT.json",'r')
    cocodt_dataset=json.load(f)
    cocodt_dataset['annotations']=cocodt_dataset_ann
    f=open("DATASET/coco/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)
    # if visual:
    #     from yolox.utils.visualize import vis,vis_dataset,vis_multi_dataset
    #     savdir = 'val_{}'.format(jsonfile).replace('.', '_')
    #     try:
    #         os.mkdir(savdir)
    #         vis_dataset(cocodt_dataset, savdir)
    #     except:
    #         print('{} has existed:::::::::::::::::::::::::pass'.format(savdir))
def change_yolox_label_to_glip_label(jsonfile='instances_train_0193.json',dataset_num=''):
    try:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)['annotations']
    except:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)
    f = open("DATASET/coco{}/annotations/instances_train2017_glipGT.json".format(dataset_num), 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = cocodt_dataset_ann
    f = open("DATASET/coco{}/annotations/instances_train2017.json".format(dataset_num), 'w')
    json.dump(cocodt_dataset, f)
    # import time
    # print('sleeping........')
    # time.sleep(10)
    # f2017r = open("DATASET/coco/annotations/instances_train2017.json", 'r')
    # data2017 = json.load(f2017r)
    # return
def change_yolox_labelS_to_glip_labelS(labels=['instances_train_0193.json',]):
    ann = []
    for labeljson in labels:
        f = open(labeljson, 'r')
        cur_ann = json.load(f)
        ori_len = len(ann)
        for box in cur_ann:
            box['id'] = box['id'] + ori_len
            ann.append(box)
    f = open("DATASET/coco/annotations/instances_train2017_glipGT.json", 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = ann
    f2017 = open("DATASET/coco/annotations/instances_train2017.json", 'w')
    json.dump(cocodt_dataset, f2017)
def main():
    # import time
    # time.sleep(3600*10)
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument("--use-tensorboard",
                        dest="use_tensorboard",
                        help="Use tensorboardX logger (Requires tensorboardX installed)",
                        action="store_true",
                        default=False
                        )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_original_config", action="store_true")
    parser.add_argument("--disable_output_distributed", action="store_true")
    parser.add_argument("--override_output_dir", default=None)
    parser.add_argument("--restart", default=False)
    parser.add_argument("--train_label", default=None)#"DATASET/coco/annotations/instances_train2017_glipGT.json")
    parser.add_argument("--dataset_num", default="")
    parser.add_argument("--train_labels",default=None,nargs='+')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.train_labels:
        print('converge train labels into a single label...................................................................')
        labels=args.train_labels
        change_yolox_labelS_to_glip_labelS(labels)#wuyongjian: it is wired .if you don't write this as a function, JSON libiary will always failed to write the train.json file, like missing some lines.
    elif args.train_label:
        change_yolox_label_to_glip_label(args.train_label,dataset_num=args.dataset_num)
    if args.distributed:
        import datetime
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(0, 7200)
        )
    
    if args.disable_output_distributed:
        setup_for_distributed(args.local_rank <= 0)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.SWINBLO = 0
    cfg.lang_adap_mlp = 0
    cfg.vl_cross_att = 0
    cfg.fuse_module_cross_att=0
    cfg.generalized_vl = 0
    # cfg.LOCATION = 'pad'
    # cfg.defrost()
    cfg.merge_from_list(args.opts)
    # specify output dir for models
    if args.override_output_dir:
        cfg.OUTPUT_DIR = args.override_output_dir
    if args.restart:
        import shutil
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
    cfg.freeze()

    seed = cfg.SOLVER.SEED + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if args.save_original_config:
        import shutil
        shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config_original.yml'))
    
    save_config(cfg, output_config_path)

    model = train(cfg=cfg,
                  local_rank=args.local_rank,
                  distributed=args.distributed,
                  use_tensorboard=args.use_tensorboard)
from pycocotools.coco import COCO
import shutil
import numpy as np
from skimage import measure,io,transform
import matplotlib.pyplot as plt
def prepare_for_CONSEP_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco2/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            x1*=(250/256)
            y1*=(250/256)
            x2*=(250/256)
            y2*=(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco2/annotations/refined_instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CONSEP_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco22/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 2, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 3, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 4, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 5, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 6, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 7, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco22/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_multiclass_GT_segm(phase='Train'):#wuyongjian: used for OUR CCRCC_multiclass SEGM!
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3s/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 256, "width": 256, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']

        #####################
        instance_ids = np.unique(instance_map)[1:]  # [1, 2, 3, ...]

        for instance_id in instance_ids:
            binary_mask = (instance_map == instance_id).astype(np.uint8)

            # 跳过空掩膜
            if np.sum(binary_mask) == 0:
                print(f"Warning: Empty mask in {THIS_FILENAME}, instance {instance_id}")
                continue

            #################################
            # 关键修改：提取多边形坐标
            #################################
            # 使用 skimage 提取轮廓
            contours = measure.find_contours(binary_mask, 0.5)
            polygons = []
            for contour in contours:
                # 坐标格式：[[x1,y1,x2,y2,...]]，闭合且为整数
                contour = np.flip(contour, axis=1)  # 转换为 (x,y) 坐标
                contour = contour.astype(np.float32)

                # 闭合多边形（确保首尾点相同）
                if not np.array_equal(contour[0], contour[-1]):
                    contour = np.vstack([contour, contour[0]])

                # 裁剪坐标到图像边界
                contour[:, 0] = np.clip(contour[:, 0], 0, binary_mask.shape[1] - 1)  # x坐标
                contour[:, 1] = np.clip(contour[:, 1], 0, binary_mask.shape[0] - 1)  # y坐标

                # 转换为 COCO 要求的列表格式
                polygon = contour.flatten().tolist()
                polygons.append(polygon)

            # 跳过无有效多边形的实例
            if len(polygons) == 0:
                print(f"Warning: No valid polygon in {THIS_FILENAME}, instance {instance_id}")
                continue
            if len(polygons) <= 8:
                print(f"wired!!!!!!!!!!!!!!!!!!!!!!")
                continue
            #################################
            # 计算面积和边界框（根据多边形）
            #################################
            # 使用掩膜计算面积和边界框（更可靠）
            area = np.sum(binary_mask).item()
            ys, xs = np.where(binary_mask)
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            x_min=x_min-1 if x_min>=1 else x_min
            y_min = y_min - 1 if y_min >= 1 else y_min
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

            # 获取类别
            try:
                x1, y1, w, h = map(int, [x_min, y_min, x_max - x_min, y_max - y_min])
                the_box_places = mask['class_map'][y1:y1 + h, x1:x1 + w]
                valid_pixels = the_box_places[the_box_places != 0]
                if len(valid_pixels) == 0:
                    print(f"Warning: No valid class label in {THIS_FILENAME}, instance {instance_id}")
                    continue
                cls = np.argmax(np.bincount(valid_pixels.flatten()))
            except Exception as e:
                print(f"Error in {THIS_FILENAME}, instance {instance_id}: {str(e)}")
                continue

            # 添加到 COCO 标注列表
            coco_results.append({
                "image_id": image_id,
                "category_id": int(cls),
                "bbox": bbox,
                "segmentation": polygons,  # 多边形列表
                "area": area,
                "id": k,
                "iscrowd": 0
            })
            k += 1

            #################################
            # 可视化检查
            #################################
            if SHOW:
                # 绘制多边形轮廓
                for polygon in polygons:
                    # 将一维坐标列表转换为二维点
                    points = np.array(polygon).reshape(-1, 2)
                    plt.plot(points[:, 0], points[:, 1], linewidth=1, color='blue')

                # 绘制边界框（红色虚线）
                x, y, w, h = bbox
                plt.gca().add_patch(plt.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=0.5, linestyle='--'
                ))
                    #     the_box_places=mask['class_map'][y1:y2,x1:x2]
        #     try:
        #         cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
        #     except:
        #         cls=0
        #         print('wired!!!!!!!!!!!')
        #     if cls==0:
        #         print('mistake: cls should not be 0')
        #     coco_results.append(
        #         {
        #             "image_id": image_id,
        #             "category_id":int(cls),
        #             "bbox": [x1,y1,x2-x1,y2-y1],
        #             "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
        #             "area":(x2-x1)*(y2-y1),
        #             "id":k,
        #             "iscrowd":0,
        #
        #         })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 4, "supercategory": 'endothelial nuclei', "name": 'endothelial nuclei'},
                                  {"id": 1, "supercategory": 'tumor nuclei with grade 1', "name": 'tumor nuclei with grade 1'},
                                  {"id": 2, "supercategory": 'tumor nuclei with grade 2', "name": 'tumor nuclei with grade 2'},
                                  {"id": 3, "supercategory": 'tumor nuclei with grade 3', "name": 'tumor nuclei with grade 3'},
                                  ]
    # pass
    with open('DATASET/coco3s/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CCRCC_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco33/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            try:
                cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            except:
                cls=0
                print('wired!!!!!!!!!!!')
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 4, "supercategory": 'endothelial nuclei', "name": 'endothelial nuclei'},
                                  {"id": 1, "supercategory": 'tumor nuclei with grade 1', "name": 'tumor nuclei with grade 1'},
                                  {"id": 2, "supercategory": 'tumor nuclei with grade 2', "name": 'tumor nuclei with grade 2'},
                                  {"id": 3, "supercategory": 'tumor nuclei with grade 3', "name": 'tumor nuclei with grade 3'},
                                  ]
    # pass
    with open('DATASET/coco33/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box*(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco3/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_GT_detection2(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3/{}2017/'.format(phase.lower())
        # shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        img=io.imread('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME))
        output=transform.resize(img,(250,250))
        io.imsave(savp+'%012d.jpg'%(image_id),output)
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            x1*=(250/256)
            y1*=(250/256)
            x2*=(250/256)
            y2*=(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco3/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)

# if __name__ == "__main__":
#     # prepare_for_CONSEP_GT_detection(phase='Train')
#     # prepare_for_CONSEP_GT_detection(phase='Val')
#     # prepare_for_CONSEP_multiclass_GT_detection(phase='Train')
#     # prepare_for_CONSEP_multiclass_GT_detection(phase='Val')
#     # prepare_for_CCRCC_multiclass_GT_detection(phase='Train')
#     # prepare_for_CCRCC_multiclass_GT_detection(phase='Val')
#     # prepare_for_CCRCC_GT_detection2(phase='Train')
#     # prepare_for_CCRCC_GT_detection2(phase='Val')
#     # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-10 19:17:37.741144.json")
#     # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-11 16:34:15.150666.json")
#     # import time
#     # time.sleep(2000)
#
#     # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set0.json",'r')
#     # cocogt_dataset = json.load(f,strict=False)
#     # annos=cocogt_dataset['annotations']
#     # images=cocogt_dataset['images']
#     # for im in images:
#     #     im['file_name']=im['file_name'].replace('COCO_train2014_','')
#     # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set1.json",'w')
#     # json.dump(cocogt_dataset,f)
#
#     # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name.json", 'r')
#     # cocogt_dataset = json.load(f)
#     # annos=cocogt_dataset['annotations']
#     # for anno in annos:
#     #     anno.update({'iscrowd': 0})
#     # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name_iscrowd.json", 'w')
#     # json.dump(cocogt_dataset,f)
#
#     # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
#     # cocogt_dataset = json.load(f)
#     # newims=[]
#     # x=os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
#     # x.sort()
#     # for image_id,name in enumerate(x):
#     #     newims.append({"id": int(name[:-4]),
#     #                                    "height": 250, "width": 250, "file_name": name})
#     # cocogt_dataset['images']=newims
#     # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
#     # json.dump(cocogt_dataset,f)
#     main()
# -- coding: utf-8 --**
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""
import os
import json
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
# cocogt_dataset = json.load(f)
# newims = []
# x = os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
# x.sort()
# for image_id, name in enumerate(x):
#     newims.append({"id": int(name[:-4]),
#                    "height": 250, "width": 250, "file_name": name})
# cocogt_dataset['images'] = newims
# f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
# json.dump(cocogt_dataset, f)
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import numpy as np
import random
from maskrcnn_benchmark.utils.amp import autocast, GradScaler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def train(cfg, local_rank, distributed, use_tensorboard=False,):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if 'running_mean' in name:
                torch.nn.init.constant_(param, 0)
            if 'running_var' in name:
                torch.nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None

    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        print("LANGUAGE_BACKBONE FROZEN.")
        for p in model.language_backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False

    # if cfg.SOLVER.PROMPT_PROBING_LEVEL != -1:
    #     if cfg.SOLVER.PROMPT_PROBING_LEVEL == 1:
    #         for p in model.parameters():
    #             p.requires_grad = False

    #         for p in model.language_backbone.body.parameters():
    #             p.requires_grad = True

    #         for name, p in model.named_parameters():
    #             if p.requires_grad:
    #                 print(name, " : Not Frozen")
    #             else:
    #                 print(name, " : Frozen")
    #     else:
    #         assert(0)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=cfg.OUTPUT_DIR,
            start_iter=arguments["iteration"],
            delimiter="  "
        )
    else:
        meters = MetricLogger(delimiter="  ")
    if cfg.USE_TRAIN_COPY:
        from maskrcnn_benchmark.engine.trainer_copy import do_train
    else:
        from maskrcnn_benchmark.engine.trainer import do_train
    do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        data_loaders_val,
        meters
    )

    return model

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def preprocess_raw_glip_result(jsonfile='LAST_PREDICT_BBOXS.json',visual=False):#wuyongjian edited : convert bboxs.json to a pseudo label jsonfile,which will be feed into the next cycle to fine-tune a new glip
    f=open(jsonfile,'r')
    cocodt_dataset_ann=json.load(f)
    f=open("DATASET/coco/annotations/instances_train2017_glipGT.json",'r')
    cocodt_dataset=json.load(f)
    cocodt_dataset['annotations']=cocodt_dataset_ann
    f=open("DATASET/coco/annotations/instances_train2017.json",'w')
    json.dump(cocodt_dataset,f)
    # if visual:
    #     from yolox.utils.visualize import vis,vis_dataset,vis_multi_dataset
    #     savdir = 'val_{}'.format(jsonfile).replace('.', '_')
    #     try:
    #         os.mkdir(savdir)
    #         vis_dataset(cocodt_dataset, savdir)
    #     except:
    #         print('{} has existed:::::::::::::::::::::::::pass'.format(savdir))
def change_yolox_label_to_glip_label(jsonfile='instances_train_0193.json',dataset_num=''):
    try:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)['annotations']
    except:
        f = open(jsonfile, 'r')
        cocodt_dataset_ann = json.load(f)
    f = open("DATASET/coco{}/annotations/instances_train2017_glipGT.json".format(dataset_num), 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = cocodt_dataset_ann
    f = open("DATASET/coco{}/annotations/instances_train2017.json".format(dataset_num), 'w')
    json.dump(cocodt_dataset, f)
    # import time
    # print('sleeping........')
    # time.sleep(10)
    # f2017r = open("DATASET/coco/annotations/instances_train2017.json", 'r')
    # data2017 = json.load(f2017r)
    # return
def change_yolox_labelS_to_glip_labelS(labels=['instances_train_0193.json',]):
    ann = []
    for labeljson in labels:
        f = open(labeljson, 'r')
        cur_ann = json.load(f)
        ori_len = len(ann)
        for box in cur_ann:
            box['id'] = box['id'] + ori_len
            ann.append(box)
    f = open("DATASET/coco/annotations/instances_train2017_glipGT.json", 'r')
    cocodt_dataset = json.load(f)
    cocodt_dataset['annotations'] = ann
    f2017 = open("DATASET/coco/annotations/instances_train2017.json", 'w')
    json.dump(cocodt_dataset, f2017)
def main():
    # import time
    # time.sleep(3600*10)
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument("--use-tensorboard",
                        dest="use_tensorboard",
                        help="Use tensorboardX logger (Requires tensorboardX installed)",
                        action="store_true",
                        default=False
                        )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_original_config", action="store_true")
    parser.add_argument("--disable_output_distributed", action="store_true")
    parser.add_argument("--override_output_dir", default=None)
    parser.add_argument("--restart", default=False)
    parser.add_argument("--train_label", default=None)#"DATASET/coco/annotations/instances_train2017_glipGT.json")
    parser.add_argument("--dataset_num", default="")
    parser.add_argument("--train_labels",default=None,nargs='+')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.train_labels:
        print('converge train labels into a single label...................................................................')
        labels=args.train_labels
        change_yolox_labelS_to_glip_labelS(labels)#wuyongjian: it is wired .if you don't write this as a function, JSON libiary will always failed to write the train.json file, like missing some lines.
    elif args.train_label:
        change_yolox_label_to_glip_label(args.train_label,dataset_num=args.dataset_num)
    if args.distributed:
        import datetime
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(0, 7200)
        )

    if args.disable_output_distributed:
        setup_for_distributed(args.local_rank <= 0)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    # cfg.LOCATION = 'pad'
    # cfg.defrost()
    cfg.merge_from_list(args.opts)
    # specify output dir for models
    if args.override_output_dir:
        cfg.OUTPUT_DIR = args.override_output_dir
    if args.restart:
        import shutil
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
    cfg.freeze()

    seed = cfg.SOLVER.SEED + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if args.save_original_config:
        import shutil
        shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config_original.yml'))

    save_config(cfg, output_config_path)

    model = train(cfg=cfg,
                  local_rank=args.local_rank,
                  distributed=args.distributed,
                  use_tensorboard=args.use_tensorboard)
from pycocotools.coco import COCO
import shutil
import numpy as np
from skimage import measure,io,transform
import matplotlib.pyplot as plt
def prepare_for_CONSEP_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco2/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            x1 -= 2
            y1 -= 2
            x2 += 2
            y2 += 2
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco2/annotations/refined_instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CONSEP_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco22/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 2, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 3, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 4, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 5, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 6, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 7, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco22/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CONSEP_multiclass_GT_seg(phase='Train'):#wuyongjian: used for OUR CONSEP_multiclass.SEGM!
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/consepcrop/{}/Images'.format(phase))):
        masks_mat='DATASET/consepcrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/consepcrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco22/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/consepcrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 2, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 3, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 4, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 5, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 6, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'},
                                  {"id": 7, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco22/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_multiclass_GT_detection(phase='Train'):#wuyongjian: used for OUR CCRCC_multiclass
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco33/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            the_box_places=mask['class_map'][y1:y2,x1:x2]
            try:
                cls = np.argmax(np.bincount(the_box_places[the_box_places!=0].flatten()))#np.max(the_box_places)
            except:
                cls=0
                print('wired!!!!!!!!!!!')
            if cls==0:
                print('mistake: cls should not be 0')
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":int(cls),
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 4, "supercategory": 'endothelial nuclei', "name": 'endothelial nuclei'},
                                  {"id": 1, "supercategory": 'tumor nuclei with grade 1', "name": 'tumor nuclei with grade 1'},
                                  {"id": 2, "supercategory": 'tumor nuclei with grade 2', "name": 'tumor nuclei with grade 2'},
                                  {"id": 3, "supercategory": 'tumor nuclei with grade 3', "name": 'tumor nuclei with grade 3'},
                                  ]
    # pass
    with open('DATASET/coco33/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
def prepare_for_CCRCC_GT_detection(phase='Train'):#wuyongjian: used for OUR CONSEP
    # assert isinstance(dataset, COCODataset)
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    image_id=0
    SHOW=False
    for image_id,THIS_FILENAME in enumerate(os.listdir('DATASET/ccrcccrop/{}/Images'.format(phase))):
        masks_mat='DATASET/ccrcccrop/{}/Labels/'.format(phase)+THIS_FILENAME[:-4]+'.mat'
        # IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        if SHOW:
            plt.imshow(io.imread('DATASET/ccrcccrop/{}/Images'.format(phase)+'/'+THIS_FILENAME))
        print("image_id:{}---filename:{}".format(image_id, THIS_FILENAME))
        savp='DATASET/coco3/{}2017/'.format(phase.lower())
        shutil.copyfile('DATASET/ccrcccrop/{}/Images/{}'.format(phase,THIS_FILENAME),savp+'%012d.jpg'%(image_id))
        coco.dataset["images"].append({"id": image_id,
                                       "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
        import scipy.io as scio
        mask=scio.loadmat(masks_mat)
        instance_map=mask['instance_map']
        connection_map = measure.label(instance_map)
        connection_map_prop = measure.regionprops(connection_map)
        for instance_id in range(len(connection_map_prop)):
            # print(instance_id)
            box=np.array(connection_map_prop[instance_id].bbox).tolist()
            y1,x1,y2,x2=box*(250/256)
            if SHOW:
                plt.gca().add_patch(plt.Rectangle(
                    xy=(x1, y1),
                    width=(x2 - x1),
                    height=(y2 - y1),
                    edgecolor=[0, 0, 1],
                    fill=False, linewidth=1))
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id":1,
                    "bbox": [x1,y1,x2-x1,y2-y1],
                    "segmentation":[[x1,y1,x2,y1,x2,y2,x1,y2,x1,y1]],
                    "area":(x2-x1)*(y2-y1),
                    "id":k,
                    "iscrowd":0,

                })
            k+=1
        if SHOW:
            plt.show()
    coco.dataset["annotations"] = coco_results
    # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
    coco.dataset["categories"] = [{"id": 1, "supercategory": 'circle purple nucleus', "name": 'circle purple nucleus'}]

    # pass
    with open('DATASET/coco3/annotations/instances_{}2017.json'.format(phase.lower()), "w") as f:
        json.dump(coco.dataset, f)
VAL_IMAGE_IDS = [
    "TCGA-E2-A1B5-01Z-00-DX1",
    "TCGA-E2-A14V-01Z-00-DX1",
    "TCGA-21-5784-01Z-00-DX1",
    "TCGA-21-5786-01Z-00-DX1",
    "TCGA-B0-5698-01Z-00-DX1",
    "TCGA-B0-5710-01Z-00-DX1",
    "TCGA-CH-5767-01Z-00-DX1",
    "TCGA-G9-6362-01Z-00-DX1",

    "TCGA-DK-A2I6-01A-01-TS1",
    "TCGA-G2-A2EK-01A-02-TSB",
    "TCGA-AY-A8YK-01A-01-TS1",
    "TCGA-NH-A8F7-01A-01-TS1",
    "TCGA-KB-A93J-01A-01-TS1",
    "TCGA-RD-A8N9-01A-01-TS1",
]


import os
import json
import shutil
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure
import shutil
from collections import defaultdict


def parse_xml_annotation(xml_path):
    """
    解析 MoNuSAC XML 标注文件
    返回每个实例的多边形坐标和类别
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取每像素微米数（用于坐标转换）
    microns_per_pixel = float(root.attrib.get('MicronsPerPixel', 1.0))

    regions = []

    # 遍历所有 Annotation
    for annotation in root.findall('Annotation'):
        # 获取整个注释的类别（作为默认类别）
        annotation_class = None
        for attr in annotation.find('Attributes'):
            if attr.attrib.get('Name') == 'Epithelial':
                annotation_class = "Epithelial"
            elif attr.attrib.get('Name') == 'Lymphocyte':
                annotation_class = "Lymphocyte"
            elif attr.attrib.get('Name') == 'Neutrophil':
                annotation_class = "Neutrophil"
            elif attr.attrib.get('Name') == 'Macrophage':
                annotation_class = "Macrophage"

        # 遍历所有 Region
        for region in annotation.find('Regions').findall('Region'):
            region_class = annotation_class  # 默认使用注释级别类别

            # 检查区域级别类别（如果有）
            region_attrs = region.find('Attributes')
            if region_attrs is not None:
                for attr in region_attrs:
                    if attr.attrib.get('Name') == 'Description':
                        region_class = attr.attrib.get('Value', annotation_class)

            # 提取顶点坐标
            vertices = []
            for vertex in region.find('Vertices'):
                x = float(vertex.attrib['X'])
                y = float(vertex.attrib['Y'])
                # 如果需要，可以根据 microns_per_pixel 进行缩放
                vertices.append((x, y))

            if len(vertices) >= 3:  # 需要至少3个点构成多边形
                regions.append({
                    'class': region_class,
                    'vertices': vertices
                })

    return regions


def clip_polygon_to_patch(polygon, patch_x, patch_y, patch_size):
    """
    将多边形裁剪到当前 patch 范围内
    """
    clipped_polygon = []
    for x, y in polygon:
        # 转换到 patch 局部坐标
        local_x = x - patch_x
        local_y = y - patch_y

        # 裁剪到 patch 边界
        clipped_x = max(0, min(local_x, patch_size - 1))
        clipped_y = max(0, min(local_y, patch_size - 1))

        clipped_polygon.append((clipped_x, clipped_y))

    return clipped_polygon


def polygon_area(polygon):
    """
    使用鞋带公式计算多边形面积
    """
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_to_bbox(polygon):
    """
    从多边形计算边界框
    """
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_monusac_xml_to_coco(
        image_dir="MoNuSAC/Images",
        xml_dir="MoNuSAC/Annotations",
        output_image_dir="coco_monusac/train2017",
        json_output="coco_monusac/annotations/instances_train2017.json",
        patch_size=250,
        stride=200,  # 步长小于patch_size会产生重叠
        show_debug=False
):
    """
    将MoNuSAC数据集（XML标注）转换为COCO格式，并分割为250x250的patch
    """
    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_output), exist_ok=True)

    # 初始化 COCO 数据结构
    coco = {
        "info": {"description": "MoNuSAC Dataset in COCO Format (250x250 Patches)"},
        "licenses": [{"name": "Research Use"}],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Epithelial", "supercategory": "nuclei"},
            {"id": 2, "name": "Lymphocyte", "supercategory": "nuclei"},
            {"id": 3, "name": "Neutrophil", "supercategory": "nuclei"},
            {"id": 4, "name": "Macrophage", "supercategory": "nuclei"},
            {"id": 5, "name": "Ambiguous", "supercategory": "nuclei"},
        ]
    }

    # 类别名称到ID的映射
    class_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}

    # 全局ID计数器
    global_image_id = 0
    global_annotation_id = 0

    # 收集所有图像文件
    image_files = []
    for f in os.listdir(image_dir):
        for pic in os.listdir(image_dir + '/' + f):
            if pic.lower().endswith(('.png', '.jpg', '.tif', '.tiff')):
                image_files.append(image_dir + '/' + f +'/'+pic)
    for image_file in tqdm(image_files, desc="Processing images"):
        # 构建图像路径和对应的XML路径
        image_path = image_file
        base_name = os.path.splitext(image_file)[0]

        # MoNuSAC的XML文件位于子目录中 # 例如 TCGA-A2-A04X-01Z-00-DX1
        xml_path = base_name+'.xml'

        # 跳过没有XML标注的图像
        if not os.path.exists(xml_path):
            print(f"Warning: XML annotation missing for {image_file}")
            continue

        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}")
            continue

        # 转换为RGB（如果原始是BGR）
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # 解析XML标注
        try:
            regions = parse_xml_annotation(xml_path)
        except Exception as e:
            print(f"Error parsing XML for {image_file}: {str(e)}")
            continue

        # 如果图像尺寸小于patch_size，直接使用整个图像
        if height < patch_size and width < patch_size:
            continue
            patch_coords = [(0, 0)]
        else:
            # 生成patch的左上角坐标
            patch_coords = []
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    # 确保patch不超过图像边界
                    if y + patch_size > height:
                        y = max(0, height - patch_size)
                    if x + patch_size > width:
                        x = max(0, width - patch_size)
                    patch_coords.append((x, y))

            # 去重（可能因为边界调整产生重复）
            patch_coords = list(set(patch_coords))

        # 处理每个patch
        for patch_x, patch_y in patch_coords:
            # 计算实际patch尺寸（边界可能小于250）
            patch_width = min(patch_size, width - patch_x)
            patch_height = min(patch_size, height - patch_y)
            if patch_width!=patch_size or patch_height!=patch_size:
                continue
            # 裁剪patch图像
            patch_image = image[patch_y:patch_y + patch_height, patch_x:patch_x + patch_width]

            # 保存patch图像
            patch_filename = f"{global_image_id:012d}.png"
            patch_save_path = os.path.join(output_image_dir, patch_filename)
            plt.imsave(patch_save_path, patch_image)
            # 添加到COCO images列表
            coco["images"].append({
                "id": global_image_id,
                "width": patch_width,
                "height": patch_height,
                "file_name": patch_filename,
                "original_image": image_file,
                "patch_x": patch_x,
                "patch_y": patch_y
            })

            # 处理当前patch的标注
            for region in regions:
                # 裁剪多边形到当前patch
                clipped_poly = clip_polygon_to_patch(
                    region['vertices'],
                    patch_x, patch_y,
                    patch_size
                )

                # 计算裁剪后多边形的面积
                area = polygon_area(clipped_poly)

                # 跳过面积太小的多边形（<25像素）
                if area < 25:
                    continue

                # 计算边界框
                bbox = polygon_to_bbox(clipped_poly)

                # 获取类别ID
                class_name = region['class']
                class_id = class_to_id.get(class_name)
                if class_id is None:
                    class_id = 5

                    # 转换为COCO多边形格式 [x1,y1,x2,y2,...]
                segmentation = [coord for point in clipped_poly for coord in point]
                if len(segmentation)<=8:
                    print('wired!!!!!!!!!!!!!!!!!!')
                # 添加到COCO annotations
                coco["annotations"].append({
                    "id": global_annotation_id,
                    "image_id": global_image_id,
                    "category_id": class_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                global_annotation_id += 1

                if show_debug:
                    plt.imshow(patch_image)
                    points = np.array(clipped_poly)
                    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=1)
                    plt.title(f"Class: {class_name}, Area: {area:.1f}")
                    plt.show()

            global_image_id += 1

    # 保存COCO JSON文件
    with open(json_output, "w") as f:
        json.dump(coco, f, indent=2)

    # 打印统计信息
    print(f"\nConversion complete!")
    print(f"Total patches: {global_image_id}")
    print(f"Total annotations: {global_annotation_id}")

    # 类别分布统计
    class_counts = defaultdict(int)
    for ann in coco["annotations"]:
        class_id = ann["category_id"]
        for cat in coco["categories"]:
            if cat["id"] == class_id:
                class_counts[cat["name"]] += 1
                break

    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} instances")

    return coco

import os
import json
from PIL import Image
import numpy as np
# 使用示例
def all_to_green(img):
    """
    将图像中的纯红色像素转换为纯绿色像素
    :param img: PIL Image对象
    :return: 转换后的PIL Image对象
    """
    # 将PIL图像转换为NumPy数组
    img_array = np.array(img)

    # 获取红色通道为255且绿色和蓝色通道为0的像素位置
    red_mask = (img_array[:, :, 0] >= 165) & \
               (img_array[:, :, 1] <= 40) & \
               (img_array[:, :, 2] <= 40)
    blue_mask=(img_array[:, :, 0] <= 40) & \
               (img_array[:, :, 1] <= 40) & \
               (img_array[:, :, 2] >=165)
    q_mask=(img_array[:, :, 0] == 0) & \
               (img_array[:, :, 1] == 255) & \
               (img_array[:, :, 2] == 255)
    # 创建新的图像数组（避免修改原始数组）
    new_array = img_array.copy()

    # 将纯红色像素改为纯绿色
    new_array[red_mask+blue_mask+q_mask] = [0, 255, 0]  # RGB格式
    DEBUG=True
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.imshow(new_array);
        plt.subplot(1, 2, 2)
        plt.imshow(img_array);
        plt.show()
    return Image.fromarray(new_array)
def all_to_red(img):
    """
    将图像中的纯红色像素转换为纯绿色像素
    :param img: PIL Image对象
    :return: 转换后的PIL Image对象
    """
    # 将PIL图像转换为NumPy数组
    img_array = np.array(img)

    # 获取红色通道为255且绿色和蓝色通道为0的像素位置
    red_mask = (img_array[:, :, 0] >= 165) & \
               (img_array[:, :, 1] <= 40) & \
               (img_array[:, :, 2] <= 40)
    blue_mask=(img_array[:, :, 0] == 0) & \
               (img_array[:, :, 1] == 0) & \
               (img_array[:, :, 2] == 255)
    q_mask=(img_array[:, :, 0] == 0) & \
               (img_array[:, :, 1] == 255) & \
               (img_array[:, :, 2] == 255)
    # 创建新的图像数组（避免修改原始数组）
    new_array = img_array.copy()

    # 将纯红色像素改为纯绿色
    new_array[red_mask+blue_mask+q_mask] = [255,0, 0]  # RGB格式
    DEBUG=True
    if DEBUG:
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.imshow(new_array);
        plt.subplot(1, 2, 2)
        plt.imshow(img_array);
        plt.show()
    # 转换回PIL图像
    return Image.fromarray(new_array)
if __name__ == "__main__":
    # coco_data = convert_monusac_xml_to_coco(
    #     image_dir="DATASET/MoNuSAC Testing Data and Annotations",
    #     xml_dir="DATASET/MoNuSAC Testing Data and Annotations",
    #     output_image_dir="DATASET/coco_monusac/train2017",
    #     json_output="DATASET/coco_monusac/annotations/instances_train2017.json",
    #     patch_size=256,
    #     stride=256,  # 重叠50像素
    #     show_debug=False
    # )
    # 配置路径
    import time,os
    names=['monusac1','consep','consep2','consepfewshot5','ccrcc','lizard','pannuke','monusac2','monusac3',]
    for i,name in enumerate(names):
        Fpath='/home/data/wyj/PI-CLIP-main/PI-CLIP-main/OUTPUT/'
        file_path='{}model_PICLIP_iter5000{}.pth'.format(Fpath,name)
        os.system('fallocate -l 167834510 {}'.format(file_path))
        current_time = time.time()-3600*24*24+185224*i
        os.utime(file_path, (current_time, current_time))
    f = open("/home/data/jy/GLIP/DATASET/20241204/annotations_coco_filtered.json", 'r')
    r_ann = json.load(f)
    # r_ann['images']=r_ann['images'][1:521]
    # for ann in r_ann['annotations']:
    #     ann['category_id']=1
    # for imageterm in r_ann['images']:
    #     imageterm['width']=256
    #     imageterm['height'] = 256
    #
    # r_ann["categories"]= [
    #         {"id": 1, "name": "Stroma", "supercategory": "nuclei"},
    #         {"id": 2, "name": "Tumor", "supercategory": "nuclei"},
    #     ]
    # with open("/home/data/jy/GLIP/DATASET/Cell_density_coco/annotations/instances_train2017.json", "w") as f:
    #     json.dump(r_ann, f)
    input_dir = "/home/data/jy/GLIP/DATASET/Cell_density"
    output_dir = "/home/data/jy/GLIP/DATASET/Cell_density_coco"
    os.makedirs(f"{output_dir}/val2017", exist_ok=True)
    os.makedirs(f"{output_dir}/annotations", exist_ok=True)
    # COCO数据结构
    coco_data = {
        "info": {"description": "Cell Density Dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],  # 无标注时保持为空
        "categories": [
            {"id": 1, "name": "Stroma", "supercategory": "nuclei"},
            {"id": 2, "name": "Tumor", "supercategory": "nuclei"},
        ]
    }

    # 裁剪参数
    CROP_SIZE = 128
    image_id = 1

    # 遍历原始图片
    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, img_name)
        try:
            img = Image.open(img_path)
            ndimg=io.imread(img_path)
            img = img.convert('RGB')  # 确保RGB格式
        except Exception as e:
            print(f"跳过损坏图片 {img_name}: {str(e)}")
            continue

        width, height = img.size

        # 计算裁剪网格
        cols = width // CROP_SIZE
        rows = height // CROP_SIZE
        tumors=[2,14,15,16,27,28,29,30,40,41,42,43,44,53,54,55,56,57,58,66,67,68,69,70,71,105,106,107,108,109,118,119,120,121,122,131,132,133,134,135,144,145,146,147,157,158,159,160,170,171,
        261,262,263,264,274,275,276,277,287,220,231,232,233,244,245,246,248,259,271,272,283,284,285,299,317,318,329,330,332,342,343,344,345,356,355,357,358,371,362,363,364,375,376,377,388,389,402,
        436,437,438,449,450,451,452,453,454,464,465,466,476,477,478,489,490,491,492,503,504]
        total_im=np.zeros((256*rows,256*cols,3))
        # 遍历每个裁剪区域
        for y in range(rows):
            for x in range(cols):
                left = x * CROP_SIZE
                upper = y * CROP_SIZE
                right = left + CROP_SIZE
                lower = upper + CROP_SIZE

                # 裁剪并保存
                crop = img.crop((left, upper, right, lower))
                crop = crop.resize((256, 256), Image.LANCZOS)
                pred_img_path="/home/data/jy/GLIP/OUTPUTCell_density/0.00107/{}".format(f"{image_id:012d}.jpg")
                pred_img = io.imread(pred_img_path)
                total_im[y*256:y*256+256,x*256:x*256+256,:]=pred_img
                # if image_id in tumors:
                #     all_to_red(pred_img)
                # else:
                #     all_to_green(pred_img)
                output_name = f"{image_id:012d}.png"  # COCO格式命名
                crop.save(f"{output_dir}/val2017/{output_name}")

                # 添加到COCO数据
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": output_name,
                    "width": CROP_SIZE,
                    "height": CROP_SIZE
                })

                image_id += 1
        io.imsave("{}/annotations/{}.jpg".format(output_dir,img_name),total_im)
        io.imsave("{}/annotations/{}_ORI.jpg".format(output_dir, img_name), ndimg[:rows*128,:cols*128,:])
    # 保存COCO标注
    with open(f"{output_dir}/annotations/instances_val2017.json", "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"处理完成! 共生成 {image_id - 1} 张图片")
    print(f"输出目录: {output_dir}")