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
def prepare_for_MONU_GT_detect_and_seg(generate_phase):#wuyongjian: used for OUR MONUSAC#CAUTION!:this function used for glip-adapter train,with key filen_name been changed
    # assert isinstance(dataset, COCODataset)
    debug=False
    print('generating GT bbox')
    coco = COCO()
    coco.dataset = {}
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco_results = []
    #for image_id, prediction in enumerate(predictions):
    k=0
    for image_id in range(480):
        FILENAME_LIST=os.listdir('/home/data/jy/GLIP/DATASET/MoNuSACGT/stage1_train/')
        FILENAME_LIST.sort()
        THIS_FILENAME=FILENAME_LIST[image_id//16]
        THIS_CROP_NUM=image_id%16
        THE_X=THIS_CROP_NUM%4
        THE_Y = THIS_CROP_NUM //4
        masks_dir='/home/data/jy/GLIP/DATASET/MoNuSACGT/stage1_train/'+THIS_FILENAME+'/masks/'
        crop_size=250
        original_id = THIS_FILENAME+'_crop_{}.png'.format(THIS_CROP_NUM)
        # if debug:
        #     IMO=io.imread('/data1/wyj/M/datasets/MoNuSACCROP/images/'+original_id)
        #     plt.imshow(IMO)
        print("image_id:{}---filename:{}".format(image_id, original_id))
        if THIS_FILENAME in VAL_IMAGE_IDS:
            phase='val'
        else:
            phase='train'
        if phase in generate_phase:
            # shutil.copyfile('/data1/wyj/M/datasets/MoNuSACCROP/images/{}'.format(original_id),'/data1/wyj/M/datasets/COCO2/%s2017/%012d.jpg'%(phase,image_id))
            coco.dataset["images"].append({"id": image_id,
                                           "height": 250, "width": 250, "file_name":'%012d.jpg'%(image_id)})
            for instance_mask_file in os.listdir(masks_dir):
                instance_im=io.imread(masks_dir+instance_mask_file)
                BORROW_PLACE = instance_im[THE_X * crop_size:(THE_X + 1) * crop_size,THE_Y * crop_size:(THE_Y + 1) * crop_size]
                if debug:
                    IMO=io.imread('/home/data/jy/GLIP/DATASET/MoNuSACGT/stage1_train/'+THIS_FILENAME+'/images/'+THIS_FILENAME+'.png')
                    plt.imshow(IMO[THE_X * crop_size:(THE_X + 1) * crop_size,THE_Y * crop_size:(THE_Y + 1) * crop_size])
                if np.max(BORROW_PLACE)>0:
                    connection_map=measure.label(BORROW_PLACE)
                    connection_map_prop=measure.regionprops(connection_map)
                    box=np.array(connection_map_prop[0].bbox).tolist()
                    y1,x1,y2,x2=box
                    contours = measure.find_contours(BORROW_PLACE)
                    segmentation=[]
                    for contour in contours:
                        segmentation += contour[:,::-1].flatten().tolist()
                        if debug:
                            plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=0.5)
                    x1-=2
                    y1-=2
                    # x2+=2
                    # y2+=2
                    if debug:
                        plt.gca().add_patch(plt.Rectangle(
                            xy=(x1, y1),
                            width=(x2 - x1),
                            height=(y2 - y1),
                            edgecolor=[0, 0, 1],
                            fill=False, linewidth=0.5))
                    if len(segmentation)<8 or len(segmentation)%2!=0:
                        print('wrong')
                        continue
                    coco_results.append(
                        {
                            "image_id": image_id,
                            "category_id":1,
                            "bbox": [x1,y1,x2-x1,y2-y1],
                            "segmentation":[segmentation],
                            "area":(x2-x1)*(y2-y1),
                            "id":k,
                            "iscrowd":0,

                        })
                    k+=1
                    if debug:
                        plt.show()
                    pass
            coco.dataset["annotations"] = coco_results
            # coco.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(classes)]
            coco.dataset["categories"] = [{"id": 1, "supercategory": 'nucleus', "name": 'nucleus'}]

    with open('DATASET/coco1/annotations/instances_{}2017.json'.format(generate_phase), "w") as f:
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
def SPLIT_VOC_TRAIN():
    f = open("DATASET/odinw/PascalVOC/train/annotations_without_background.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    selected_class_set=[['bird','bus','cow','motorbike','sofa'],['aeroplane','bottle','cow','horse','sofa'],['boat','cat','motorbike','sheep','sofa']]
    for i,set in enumerate(selected_class_set):
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            if categories[this_cat_id-1]['name'] not in set:
                coco.dataset["annotations"].append(ann)
                coco.dataset["images"][this_image_id-1]['license']=0
                coco.dataset["categories"][this_cat_id - 1]['supercategory'] = 'SPLIT_{}'.format(i)
        for id in range(len(coco.dataset["images"])-1,-1,-1):
            image=coco.dataset["images"][id]
            if image['license']!=0:
                coco.dataset["images"].pop(id)
        for id in range(len(coco.dataset["categories"])-1,-1,-1):
            sc=coco.dataset["categories"][id]
            if sc['supercategory']!='SPLIT_{}'.format(i):
                coco.dataset["categories"].pop(id)
        f=open("DATASET/odinw/PascalVOC/train/SPLIT_{}.json".format(i),'w')
        json.dump(coco.dataset,f)
def SPLIT_VOC_VAL_hard():
    f = open("DATASET/odinw/PascalVOC/valid/annotations_without_background.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    selected_class_set=[['bird','bus','cow','motorbike','sofa'],['aeroplane','bottle','cow','horse','sofa'],['boat','cat','motorbike','sheep','sofa']]
    for i,set in enumerate(selected_class_set):
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            if categories[this_cat_id-1]['name'] in set:
                coco.dataset["annotations"].append(ann)
                coco.dataset["images"][this_image_id-1]['license']=0
                coco.dataset["categories"][this_cat_id - 1]['supercategory'] = 'SPLIT_{}'.format(i)
        for id in range(len(coco.dataset["images"])-1,-1,-1):
            image=coco.dataset["images"][id]
            if image['license']!=0:
                coco.dataset["images"].pop(id)
        for id in range(len(coco.dataset["categories"])-1,-1,-1):
            sc=coco.dataset["categories"][id]
            if sc['supercategory']!='SPLIT_{}'.format(i):
                coco.dataset["categories"].pop(id)
        f=open("DATASET/odinw/PascalVOC/valid/SPLIT_hard{}.json".format(i),'w')
        json.dump(coco.dataset,f)
def SPLIT_VOC_VAL():
    f = open("DATASET/odinw/PascalVOC/valid/annotations_without_background.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    selected_class_set=[['bird','bus','cow','motorbike','sofa'],['aeroplane','bottle','cow','horse','sofa'],['boat','cat','motorbike','sheep','sofa']]
    for i,set in enumerate(selected_class_set):
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            if categories[this_cat_id-1]['name'] in set:
                coco.dataset["annotations"].append(ann)
                coco.dataset["images"][this_image_id-1]['license']=0
                coco.dataset["categories"][this_cat_id - 1]['supercategory'] = 'SPLIT_{}'.format(i)
        # for id in range(len(coco.dataset["images"])-1,-1,-1):
        #     image=coco.dataset["images"][id]
        #     if image['license']!=0:
        #         coco.dataset["images"].pop(id)
        for id in range(len(coco.dataset["categories"])-1,-1,-1):
            sc=coco.dataset["categories"][id]
            if sc['supercategory']!='SPLIT_{}'.format(i):
                coco.dataset["categories"].pop(id)
        f=open("DATASET/odinw/PascalVOC/valid/SPLIT_{}.json".format(i),'w')
        json.dump(coco.dataset,f)
def SPLIT_COCO_VAL():
    f = open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_val2017.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    selected_class_set=[['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motorcycle','person','potted plant','sheep','couch','train','tv']]
    for i,set in enumerate(selected_class_set):
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            cat_name=''
            for cat in coco.dataset["categories"]:
                if cat['id'] == this_cat_id:
                    cat_name=cat['name']
            if cat_name in set:
                coco.dataset["annotations"].append(ann)
                for im in coco.dataset["images"]:
                    if im['id']==this_image_id:
                        im['license']=0
                        break #only speed up a little
                for cat in coco.dataset["categories"]:
                    if cat['id']==this_cat_id:
                        cat['supercategory']='SPLIT_{}'.format(i)
                        break #only speed up a little
        # for id in range(len(coco.dataset["images"])-1,-1,-1):
        #     image=coco.dataset["images"][id]
        #     if image['license']!=0:
        #         coco.dataset["images"].pop(id)
        for id in range(len(coco.dataset["categories"])-1,-1,-1):
            sc=coco.dataset["categories"][id]
            if sc['supercategory']!='SPLIT_{}'.format(i):
                coco.dataset["categories"].pop(id)
        f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/val_SPLIT_{}.json".format(i),'w')
        json.dump(coco.dataset,f)
def SPLIT_COCO_VAL1shot(shot=1):
    f = open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_val2017.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    selected_class_set=[['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motorcycle','person','potted plant','sheep','couch','train','tv']]
    for i,set in enumerate(selected_class_set):
        selected_class_set_COUNT = np.zeros(81)
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            cat_name=''
            for cat in coco.dataset["categories"]:
                if cat['id'] == this_cat_id:
                    cat_name=cat['name']
            if cat_name in set:
                if selected_class_set_COUNT[this_cat_id]<shot:
                    coco.dataset["annotations"].append(ann)
                    selected_class_set_COUNT[this_cat_id] += 1  # once we accept a bbox(a-shot),we need to uodate class_counter
                else:
                    continue #skip this ann
                for im in coco.dataset["images"]:
                    if im['id']==this_image_id:
                        im['license']=0
                        break #only speed up a little
                for cat in coco.dataset["categories"]:
                    if cat['id']==this_cat_id:
                        cat['supercategory']='SPLIT_{}'.format(i)
                        break #only speed up a little
        for id in range(len(coco.dataset["images"])-1,-1,-1):
            image=coco.dataset["images"][id]
            if image['license']!=0:
                coco.dataset["images"].pop(id)
        for id in range(len(coco.dataset["categories"])-1,-1,-1):
            sc=coco.dataset["categories"][id]
            if sc['supercategory']!='SPLIT_{}'.format(i):
                coco.dataset["categories"].pop(id)
        f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/val_SPLIT_{}_{}shot.json".format(i,shot),'w')
        json.dump(coco.dataset,f)
def SPLIT_COCO_VAL_BASE():
    f = open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_val2017.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    all_class_set=set()
    for cid,ccat in enumerate(categories):
        all_class_set.add(ccat['name'])
    selected_class_set=[{'airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motorcycle','person','potted plant','sheep','couch','train','tv'}]
    selected_class_set[0]=list(all_class_set-selected_class_set[0])
    for i,set_ in enumerate(selected_class_set):
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            cat_name=''
            for cat in coco.dataset["categories"]:
                if cat['id'] == this_cat_id:
                    cat_name=cat['name']
            if cat_name in set_:
                coco.dataset["annotations"].append(ann)
                for im in coco.dataset["images"]:
                    if im['id']==this_image_id:
                        im['license']=0
                        break #only speed up a little
                for cat in coco.dataset["categories"]:
                    if cat['id']==this_cat_id:
                        cat['supercategory']='SPLIT_{}'.format(i)
                        break #only speed up a little
        for id in range(len(coco.dataset["images"])-1,-1,-1):
            image=coco.dataset["images"][id]
            if image['license']!=0:
                coco.dataset["images"].pop(id)
        # for id in range(len(coco.dataset["categories"])-1,-1,-1):
        #     sc=coco.dataset["categories"][id]
        #     if sc['supercategory']!='SPLIT_{}'.format(i):
        #         coco.dataset["categories"].pop(id)
        f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/val_SPLIT_baseEDIT2.json".format(i),'w')
        json.dump(coco.dataset,f)
def SPLIT_COCO_TRAIN():
    f = open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_train2017.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    selected_class_set=[['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motorcycle','person','potted plant','sheep','couch','train','tv']]
    for i,set in enumerate(selected_class_set):
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            cat_name=''
            for cat in coco.dataset["categories"]:
                if cat['id'] == this_cat_id:
                    cat_name=cat['name']
            if cat_name not in set:
                coco.dataset["annotations"].append(ann)
                for im in coco.dataset["images"]:
                    if im['id']==this_image_id:
                        im['license']=0
                        break #only speed up a little
                for cat in coco.dataset["categories"]:
                    if cat['id']==this_cat_id:
                        cat['supercategory']='SPLIT_{}'.format(i)
                        break #only speed up a little
        for id in range(len(coco.dataset["images"])-1,-1,-1):
            image=coco.dataset["images"][id]
            if image['license']!=0:
                coco.dataset["images"].pop(id)
        for id in range(len(coco.dataset["categories"])-1,-1,-1):
            sc=coco.dataset["categories"][id]
            if sc['supercategory']!='SPLIT_{}'.format(i):
                coco.dataset["categories"].pop(id)
        f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/train_SPLIT_{}.json".format(i),'w')
        json.dump(coco.dataset,f)
def SPLIT_COCO_TRAIN1shot(shot=1):
    f = open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_train2017.json", 'r')
    cocogt_dataset = json.load(f)
    newims=[]
    annotations=cocogt_dataset['annotations']
    categories=cocogt_dataset['categories']
    images=cocogt_dataset['images']
    selected_class_set=[['airplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','dining table','dog','horse','motorcycle','person','potted plant','sheep','couch','train','tv']]
    for i,set in enumerate(selected_class_set):
        selected_class_set_COUNT = np.zeros(81)
        coco = COCO()
        coco.dataset = {}
        coco.dataset["images"] = images.copy()
        coco.dataset["annotations"] = []
        coco.dataset["categories"] = categories.copy()
        for ann in annotations:
            this_cat_id=ann['category_id']
            this_image_id=ann['image_id']
            cat_name=''
            for cat in coco.dataset["categories"]:
                if cat['id'] == this_cat_id:
                    cat_name=cat['name']
            if cat_name in set:#!!!!!caution, in or not in ?
                if selected_class_set_COUNT[this_cat_id] < shot:
                    coco.dataset["annotations"].append(ann)
                    selected_class_set_COUNT[
                        this_cat_id] += 1  # once we accept a bbox(a-shot),we need to uodate class_counter
                else:
                    continue  # skip this ann
                for im in coco.dataset["images"]:
                    if im['id']==this_image_id:
                        im['license']=0
                        break #only speed up a little
                for cat in coco.dataset["categories"]:
                    if cat['id']==this_cat_id:
                        cat['supercategory']='SPLIT_{}'.format(i)
                        break #only speed up a little
        for id in range(len(coco.dataset["images"])-1,-1,-1):
            image=coco.dataset["images"][id]
            if image['license']!=0:
                coco.dataset["images"].pop(id)
        for id in range(len(coco.dataset["categories"])-1,-1,-1):
            sc=coco.dataset["categories"][id]
            if sc['supercategory']!='SPLIT_{}'.format(i):
                coco.dataset["categories"].pop(id)
        f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/train_SPLIT_{}_{}shot.json".format(i,shot),'w')
        json.dump(coco.dataset,f)

#
# import os
# import cv2
# import numpy as np
# import scipy.io as sio
# from pathlib import Path
# import json
#
#
# def split_image_and_label(image_path, label_path, output_image_dir, json_output_path):
#     os.makedirs(output_image_dir, exist_ok=True)
#
#     coco_output = {
#         "images": [],
#         "annotations": [],
#         "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 8)]
#     }
#     annotation_id = 1
#     file_counter = 1
#
#     image_files = sorted(Path(image_path).glob("*.png"))
#     for image_file in image_files:
#         image = cv2.imread(str(image_file))
#         h, w, _ = image.shape
#         assert h % 4 == 0 and w % 4 == 0, "Image dimensions must be divisible by 4"
#
#         mat_file = sio.loadmat(str(Path(label_path) / (image_file.stem + '.mat')))
#         inst_map = mat_file['inst_map']
#         type_map = mat_file['type_map']
#
#         patch_h, patch_w = h // 4, w // 4
#
#         for i in range(4):
#             for j in range(4):
#                 x_start, y_start = j * patch_w, i * patch_h
#                 x_end, y_end = x_start + patch_w, y_start + patch_h
#
#                 # Crop the image and labels
#                 cropped_image = image[y_start:y_end, x_start:x_end]
#                 cropped_inst_map = inst_map[y_start:y_end, x_start:x_end]
#                 cropped_type_map = type_map[y_start:y_end, x_start:x_end]
#
#                 # Save cropped image
#                 new_file_name = f"{file_counter:012}.jpg"
#                 cv2.imwrite(str(Path(output_image_dir) / new_file_name), cropped_image)
#
#                 # Generate COCO annotations
#                 unique_instances = np.unique(cropped_inst_map)
#                 for instance_id in unique_instances:
#                     if instance_id == 0:
#                         continue  # Skip background
#
#                     mask = (cropped_inst_map == instance_id).astype(np.uint8)
#                     # category_id = int(mat_file['inst_type'][instance_id - 1, 0])
#                     category_id = np.unique(cropped_type_map[mask == 1])
#                     if len(category_id) > 1:
#                         print(f"Warning: Instance {instance_id} has multiple types: {category_id}")
#                     category_id = int(category_id[0])
#                     # Calculate bbox (based on cropped image)
#                     y_indices, x_indices = np.where(mask)
#                     x_min, y_min = x_indices.min(), y_indices.min()
#                     x_max, y_max = x_indices.max(), y_indices.max()
#                     x_min = max(0, x_min - 2)
#                     y_min = max(0, y_min - 2)
#                     x_max = min(250, x_max + 2)
#                     y_max = min(250, y_max + 2)
#                     bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
#
#                     # Encode RLE (simplified binary mask format)
#                     # rle = {"counts": mask.flatten().tolist(), "size": [patch_h, patch_w]}
#
#                     # Add annotation
#                     coco_output["annotations"].append({
#                         "id": annotation_id,
#                         "image_id": file_counter,
#                         "category_id": category_id,
#                         "bbox": bbox,
#                         # "segmentation": rle,
#                         "iscrowd": 0,
#                         "area": bbox[2] * bbox[3]
#                     })
#                     annotation_id += 1
#
#                 # Add image info
#                 coco_output["images"].append({
#                     "id": file_counter,
#                     "file_name": new_file_name,
#                     "width": patch_w,
#                     "height": patch_h
#                 })
#                 file_counter += 1
#
#     # Save COCO JSON
#     with open(json_output_path, 'w') as f:
#         json.dump(coco_output, f)
#
#
# # 使用示例
# image_dir = "DATASET/CoNSeP/Train/Images"
# label_dir = "DATASET/CoNSeP/Train/Labels"
# output_image_dir = "DATASET/coco2s/train2017/"
# json_output = "DATASET/coco2s/annotations/instances_train2017.json"
# split_image_and_label(image_dir, label_dir, output_image_dir, json_output)

import os
import json
import shutil
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from skimage import measure
from tqdm import tqdm


def convert_consep_to_coco(
        image_dir="DATASET/CoNSeP/Train/Images",
        label_dir="DATASET/CoNSeP/Train/Labels",
        output_image_dir="DATASET/coco2s/train2017/",
        json_output="DATASET/coco2s/annotations/instances_train2017.json",
        patch_size=250,
        show_debug=True
):
    # 初始化 COCO 数据结构
    coco = {
        "info": {"description": "CoNSeP Dataset in COCO Format (250x250 Patches)"},
        "licenses": [{"name": "Research Use"}],
        "images": [],
        "annotations": [],
        "categories": [
        {'id':1, 'supercategory':'other','name':'other'},
        {'id':2, 'supercategory':'inflammatory','name':'inflammatory'},
        {'id':3, 'supercategory':'healthy epithelial','name':'healthy epithelial'},
        {'id':4, 'supercategory':'dysplastic/malignant epithelial','name':'dysplastic/malignant epithelial'},
        {'id':5, 'supercategory':'fibroblast','name':'fibroblast'},
        {'id':6, 'supercategory':'muscle','name':'muscle'},
        {'id':7, 'supercategory':'endothelial','name':'endothelial'}]
    }

    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_output), exist_ok=True)

    # 全局ID计数器
    global_image_id = 0
    global_annotation_id = 0

    # 遍历原始图像
    for filename in tqdm(os.listdir(image_dir)):
        if not filename.endswith(".png"):
            continue

        # 加载图像和标注
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".png", ".mat"))
        image = plt.imread(image_path)
        label_data = scio.loadmat(label_path)
        instance_map = label_data["inst_map"]  # 实例ID矩阵
        class_map = label_data["type_map"]  # 类别标签矩阵 (1: inflammatory, 2: epithelial, 3: spindle)

        # 获取原图尺寸 (假设为1000x1000)
        height, width = 1000, 1000

        # 分割为 250x250 Patches
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                # 裁剪 Patch 区域
                y_end = y + patch_size
                x_end = x + patch_size
                patch_image = image[y:y_end, x:x_end]
                patch_instance = instance_map[y:y_end, x:x_end]
                patch_class = class_map[y:y_end, x:x_end]

                # 跳过无效 Patch
                if patch_image.shape[0] != patch_size or patch_image.shape[1] != patch_size:
                    continue

                # 保存 Patch 图像
                patch_filename = f"{global_image_id:012d}.png"
                patch_save_path = os.path.join(output_image_dir, patch_filename)
                plt.imsave(patch_save_path, patch_image)

                # 添加到 COCO images 列表
                coco["images"].append({
                    "id": global_image_id,
                    "width": patch_size,
                    "height": patch_size,
                    "file_name": patch_filename
                })

                # 处理当前 Patch 的标注
                instance_ids = np.unique(patch_instance)[1:]  # 排除背景0
                for instance_id in instance_ids:
                    # 提取实例掩膜
                    binary_mask = (patch_instance == instance_id).astype(np.uint8)

                    # 跳过空掩膜
                    if np.sum(binary_mask) == 0:
                        continue

                    # 提取多边形轮廓
                    contours = measure.find_contours(binary_mask, 0.5)
                    polygons = []
                    for contour in contours:
                        # 坐标转换为 (x,y) 并闭合
                        contour = np.flip(contour, axis=1)
                        if not np.array_equal(contour[0], contour[-1]):
                            contour = np.vstack([contour, contour[0]])
                        # 裁剪到 Patch 边界
                        contour[:, 0] = np.clip(contour[:, 0], 0, patch_size - 1)
                        contour[:, 1] = np.clip(contour[:, 1], 0, patch_size - 1)
                        polygons.append(contour.flatten().tolist())

                    # 跳过无效实例
                    if len(polygons) == 0:
                        continue

                    # 计算面积和边界框
                    ys, xs = np.where(binary_mask)
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                    area = float(np.sum(binary_mask))

                    # 获取类别（从 class_map 中对应实例区域）
                    try:
                        x1, y1, w, h = map(int, [x_min, y_min, x_max - x_min, y_max - y_min])
                        class_region = patch_class[y1:y1 + h, x1:x1 + w]
                        valid_pixels = class_region[class_region != 0].astype(np.int64)  # 强制类型转换
                        if len(valid_pixels) == 0:
                            continue  # 跳过无有效类别的实例
                        cls = int(np.argmax(np.bincount(valid_pixels.flatten())))
                    except:
                        print('wired!!!!!!!!!!!!!!!')
                        continue

                    # 添加到 COCO annotations
                    coco["annotations"].append({
                        "id": global_annotation_id,
                        "image_id": global_image_id,
                        "category_id": cls,
                        "segmentation": polygons,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    global_annotation_id += 1

                # 可视化检查
                if show_debug:
                    plt.imshow(patch_image)
                    for ann in coco["annotations"]:
                        if ann["image_id"] == global_image_id:
                            for polygon in ann["segmentation"]:
                                points = np.array(polygon).reshape(-1, 2)
                                plt.plot(points[:, 0], points[:, 1], linewidth=1, color='blue')
                    plt.title(f"Patch {global_image_id}")
                    plt.show()

                global_image_id += 1

    # 保存 COCO JSON 文件
    with open(json_output, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    convert_consep_to_coco(
        image_dir="DATASET/CoNSeP/Train/Images",
        label_dir="DATASET/CoNSeP/Train/Labels",
        output_image_dir="DATASET/coco2s/train2017/",
        json_output="DATASET/coco2s/annotations/instances_train2017.json",
        show_debug=True
    )
    convert_consep_to_coco(
        image_dir="DATASET/CoNSeP/Test/Images",
        label_dir="DATASET/CoNSeP/Test/Labels",
        output_image_dir="DATASET/coco2s/val2017/",
        json_output="DATASET/coco2s/annotations/instances_val2017.json",
        show_debug=False
    )
# if __name__ == "__main__":
    # prepare_for_CONSEP_GT_detection(phase='Train')
    # prepare_for_CONSEP_GT_detection(phase='Val')
    # prepare_for_CONSEP_multiclass_GT_detection(phase='Train')
    # prepare_for_CONSEP_multiclass_GT_detection(phase='Val')
    # prepare_for_CCRCC_multiclass_GT_detection(phase='Train')
    # prepare_for_CCRCC_multiclass_GT_detection(phase='Val')
    # prepare_for_CCRCC_GT_detection2(phase='Train')
    # prepare_for_CCRCC_GT_detection2(phase='Val')
    # prepare_for_CCRCC_multiclass_GT_segm('Val')
    # prepare_for_CCRCC_multiclass_GT_segm('Train')
    # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-10 19:17:37.741144.json")
    # preprocess_raw_glip_result("/data2/wyj/GLIP/jsonfiles/LAST_PREDICT_BBOXS2023-07-11 16:34:15.150666.json")
    # import time
    # time.sleep(2000)
    # prepare_for_CONSEP_multiclass_GT_seg('Val')
    # prepare_for_CONSEP_multiclass_GT_seg('Train')
    # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set0.json",'r')
    # cocogt_dataset = json.load(f,strict=False)
    # annos=cocogt_dataset['annotations']
    # images=cocogt_dataset['images']
    # for im in images:
    #     im['file_name']=im['file_name'].replace('COCO_train2014_','')
    # f=open("/home/data/jy/GLIP/DATASET/coco0/annotations/instances_set1.json",'w')
    # json.dump(cocogt_dataset,f)

    # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name.json", 'r')
    # cocogt_dataset = json.load(f)
    # annos=cocogt_dataset['annotations']
    # for anno in annos:
    #     anno.update({'iscrowd': 0})
    # f = open("/data1/wyj/GLIP/DATASET/coco/annotations/lvis_v1_minival_inserted_image_name_iscrowd.json", 'w')
    # json.dump(cocogt_dataset,f)

    # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'r')
    # cocogt_dataset = json.load(f)
    # newims=[]
    # x=os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
    # x.sort()
    # for image_id,name in enumerate(x):
    #     newims.append({"id": int(name[:-4]),
    #                                    "height": 250, "width": 250, "file_name": name})
    # cocogt_dataset['images']=newims
    # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
    # json.dump(cocogt_dataset,f)
    # SPLIT_VOC_TRAIN()
    # SPLIT_VOC_VAL_BASE()
    # SPLIT_VOC_VAL()
    # SPLIT_VOC_VAL_hard()
    # SPLIT_COCO_VAL_BASE()
    # SPLIT_COCO_TRAIN()
    # SPLIT_COCO_TRAIN1shot()
    # x=os.listdir('/data2/wyj/GLIP/DATASET/coco/val2017')
    # x.sort()
    # for image_id,name in enumerate(x):
    #     newims.append({"id": int(name[:-4]),
    #                                    "height": 250, "width": 250, "file_name": name})
    # cocogt_dataset['images']=newims
    # f = open("/data2/wyj/GLIP/DATASET/coco/annotations/instances_val2017.json", 'w')
    # json.dump(cocogt_dataset,f)
    # main()