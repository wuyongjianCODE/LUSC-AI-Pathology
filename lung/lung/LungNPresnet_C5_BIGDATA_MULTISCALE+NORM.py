"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
import warnings
import openslide
warnings.filterwarnings("ignore")
from openslide.deepzoom import DeepZoomGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# import pandas as pd
from keras.models import load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten
import keras.layers as KL
from keras import metrics
from keras import backend as K
import numpy as np
from skimage import io
# from utils.visualizations import GradCAM, GuidedGradCAM, GBP
# from utils.visualizations import LRP, CLRP, LRPA, LRPB, LRPE
# from utils.visualizations import SGLRP, SGLRPSeqA, SGLRPSeqB
# from utils.helper import heatmap
# import innvestigate.utils as iutils
from skimage import io
import os,sys
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import test
from imgaug import augmenters as iaa
import tensorflow as tf
import re
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
from mrcnn import visualize
from keras import optimizers
from keras.utils import multi_gpu_model
from tensorflow.keras.utils import Sequence
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
epoch_of_current_iteration=np.array([10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
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
subset1 = [
                "TCGA-G9-6362-01Z-00-DX1",
                "TCGA-DK-A2I6-01A-01-TS1",
                "TCGA-G2-A2EK-01A-02-TSB",
                "TCGA-AY-A8YK-01A-01-TS1",
                "TCGA-NH-A8F7-01A-01-TS1",
                "TCGA-KB-A93J-01A-01-TS1",
                "TCGA-RD-A8N9-01A-01-TS1",
            ]
subset2 = [
                "TCGA-E2-A1B5-01Z-00-DX1",
                "TCGA-E2-A14V-01Z-00-DX1",
                "TCGA-21-5784-01Z-00-DX1",
                "TCGA-21-5786-01Z-00-DX1",
                "TCGA-B0-5698-01Z-00-DX1",
                "TCGA-B0-5710-01Z-00-DX1",
                "TCGA-CH-5767-01Z-00-DX1",
            ]

############################################################
#  Configurations
############################################################

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
def metric_precision(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    return precision

def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN)
    return recall

def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1score=2*precision*recall/(precision+recall)
    return F1score



############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def lab2y(label):
    y=np.zeros((5,),dtype=np.int8)
    if 'BASOPHIL' in label:
        y[0]=1
    if 'EOSINOPHIL' in label:
        y[1]=1
    if 'LYMPHOCYTE' in label:
        y[2]=1
    if 'MONOCYTE' in label:
        y[3]=1
    if 'NEUTROPHIL' in label:
        y[4]=1

    return y
def y2lab(y):
    label=[]
    if y[0]>0.2:
        label.append('Normal')
    if y[1]>0.2:
        label.append('BED')
    if y[2]>0.2:
        label.append('Tumor')
    if y[3]>0.2:
        label.append('Nacrotic')
    if y[4]>0.2:
        label.append('Normal_inside')
    print(label)
    return label
############################################################
#  Command Line
############################################################
TARGETDIR='/data1/wyj/M/samples/lung/TOSHOW7_2/'
ORIDIR='/data3/kfb'
GTDIR='/data1/wyj/M/samples/lung/TOSHOW5/'
def eval(targetdir=TARGETDIR,gtdir=GTDIR):
    GT_mpr_dict={}
    for fname in os.listdir(gtdir):
        SVSNAME=fname[:fname.find('_mpr')]
        GT_mpr=float(fname[fname.find('_mpr')+4:fname.find('.png')])
        GT_mpr_dict.update({SVSNAME : GT_mpr})
    tar_mpr_dict = {}
    for fname in os.listdir(targetdir):
        SVSNAME=fname[:fname.find('_whole_classify_mpr')]
        tar_mpr=float(fname[fname.find('_mpr')+4:fname.find('.png')])
        tar_mpr_dict.update({SVSNAME : tar_mpr})
    GT_list=list(GT_mpr_dict.keys())
    loss_values=[]
    for keyn in GT_list:
        if GT_mpr_dict[keyn]<1:
            loss_values.append(tar_mpr_dict[keyn]-GT_mpr_dict[keyn])
    mean=np.mean(loss_values)
    std=np.std(loss_values)
    print('mean : {}  std : {}'.format(mean,std))
    return mean,std
from keras_preprocessing.image.directory_iterator import DirectoryIterator
class SCALE3x3iterator(DirectoryIterator):
    def __init__(self, dir_iter):#wuyongjian:this class ,as a subclass of dir_iter,only can be init from a existed dir_iter
        super(SCALE3x3iterator,self).__init__(directory=dir_iter.directory,
                 image_data_generator=dir_iter.image_data_generator,
                 target_size=dir_iter.target_size,
                 color_mode=dir_iter.color_mode,
                 classes=None,
                 class_mode=dir_iter.class_mode,
                 batch_size=dir_iter.batch_size,
                 shuffle=dir_iter.shuffle,
                 seed=dir_iter.seed,
                 data_format=dir_iter.data_format,
                 save_to_dir=dir_iter.save_to_dir,
                 save_prefix=dir_iter.save_prefix,
                 save_format=dir_iter.save_format,
                 follow_links=False,
                 subset=None,
                 interpolation=dir_iter.interpolation,
                 dtype=dir_iter.dtype)

    # def _set_index_array(self):
    #     self.index_array = np.arange(self.n)
    #     if self.shuffle:
    #         self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        # print('check this!!!!!!!!!!!')
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + (672,672,3), dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        scale=3
        BORROW_PLACE_IM=np.zeros((scale*224,scale*224,3),dtype=np.uint8)
        for i, j in enumerate(index_array):
            for idx in [-1,0,1]:
                for y in [-1,0,1]:
                    center_imgpath=filepaths[j]
                    this_x=int(center_imgpath[center_imgpath.find('_x')+2:center_imgpath.find('_y')])+idx*224
                    this_y=int(center_imgpath[center_imgpath.find('_y')+2:center_imgpath.rfind('_')])+y*224
                    this_position_impath=center_imgpath
                    this_position_impath=this_position_impath[:center_imgpath.find('_x')+2]+str(this_x)+this_position_impath[center_imgpath.find('_y'):]
                    this_position_impath=this_position_impath[:center_imgpath.find('_y')+2]+str(this_y)+this_position_impath[center_imgpath.rfind('_'):]

                    try:
                        img = load_img(this_position_impath,
                                       color_mode=self.color_mode,
                                       target_size=self.target_size,
                                       interpolation=self.interpolation)
                        this_part = img_to_array(img, data_format=self.data_format)
                    except:
                        this_position_impath_without_NUM = this_position_impath[this_position_impath.rfind(
                            '/') + 1:this_position_impath.rfind('_')]
                        def try_to_get_im(this_position_impath_without_NUM):
                            path = r"/data3/dataL/"
                            for phasedir in os.listdir(path):
                                for classdir in os.listdir(path + phasedir):
                                    oripath = path + phasedir + '/' + classdir + '/' + this_position_impath_without_NUM
                                    try:
                                        img = load_img(this_position_impath_without_NUM,
                                                       color_mode=self.color_mode,
                                                       target_size=self.target_size,
                                                       interpolation=self.interpolation)
                                        this_part = img_to_array(img, data_format=self.data_format)
                                        return this_part
                                    except:
                                        pass
                            return np.zeros((224,224,3),dtype=np.uint8)
                        this_part = try_to_get_im(this_position_impath_without_NUM)

                    BORROW_PLACE_IM[(y+1)*224:(y+2)*224,(idx+1)*224:(idx+2)*224,:]=this_part
            x=BORROW_PLACE_IM
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]

def touch_every_img(path='/data3/dataL/'):
    count=0
    for phasedir in os.listdir(path):
        for classdir in os.listdir(path+phasedir):
            for imgpath in os.listdir(path+phasedir+'/'+classdir):
                count+=1
                oripath=path+phasedir+'/'+classdir+'/'+imgpath
                finalpath=oripath
                finalpath=oripath[:oripath.rfind('_')+1]+'.png'
                os.rename(oripath,finalpath)
                if count%1000==0:
                    print(count)
if __name__ == '__main__':
    # eval(TARGETDIR,GTDIR)
    # touch_every_img()
    import staintools
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect' or 'eval'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--iteration', required=False,
                        default=0,
                        metavar="the iteration num",
                        help='as shown')
    parser.add_argument('--LRPTS_DIR', required=False,
                        metavar="the LRPTS dir path",
                        help='as shown')
    parser.add_argument('--LOOP', required=False,
                        metavar="the LOOP number",
                        help='as shown')
    parser.add_argument('--parallel', required=False,default=0,type=int,
                        metavar="the LOOP number",
                        help='as shown')
    parser.add_argument('--plt', required=False,default=False,type=bool,
                        metavar="the LOOP number",
                        help='as shown')
    parser.add_argument('--use_norm', required=False,default=False,type=bool,
                        metavar="the LOOP number",
                        help='as shown')
    parser.add_argument('--start', required=False,default=276,
                        metavar="the LOOP number",
                        help='as shown')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)
    import matplotlib
    if not args.plt:
        # Agg backend runs without a display
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # f=open("/data1/wyj/M/datasets/dataset-master/labels.csv", "rb")

    # data = np.loadtxt(f,#delimiter=',',
     #                 skiprows=1,usecols=[1,2],dtype=object)
    # data = pd.read_csv("/data1/wyj/M/datasets/dataset-master/labels.csv",sep=',',usecols=[2])
    # print(data)
    # lab = np.array(data)[:,0]

    # Configurations
    # if args.command == "train":
    #     config = NucleusConfig()
    # else:
    #     config = NucleusInferenceConfig()
    # config.display()

    # Create model
    # if args.command == "train":
    #     Tempmodel = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs)
    # else:
    #     Tempmodel = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs)
    # alldata=[]
    # labs=[]
    # DIRP="/data1/wyj/M/datasets/dataset-master/JPEGImages"
    # for fname in sorted(os.listdir(DIRP)):
    #     im=io.imread(DIRP+'/'+fname)
    #     name = fname[fname.find('_') + 1:fname.find('.jpg')]
    #     NA=int(name)
    #     alldata.append(im)
    #     y=lab[NA]
    #     if y !=y:
    #         labs.append(lab2y([]))
    #         continue
    #     temp=y.replace(', ',',').split(',')
    #     # print(lab2y(temp))
    #     labs.append(lab2y(temp))
    # alldata=np.array(alldata)
    # labs=np.array(labs)
    model0 = InceptionResNetV2(weights='../../inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
    x=model0.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)
    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器，假设我们有2个类
    predictions = Dense(5, activation='softmax')(x)
    model = Model(inputs=model0.input, outputs=predictions)
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    if args.command == "train":
        parallel=args.parallel
    else:
        parallel=args.parallel #when detect,what to do !!!!!!!!!!!!!!!!!!!!!!!!!!
    if parallel:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=sgd,#'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=[metrics.mae, metrics.categorical_accuracy,metric_precision,metric_recall,metric_F1score,precision,recall,fmeasure,fbeta_score])
    else:
        model.compile(optimizer=sgd,#'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=[metrics.mae, metrics.categorical_accuracy,metric_precision,metric_recall,metric_F1score,precision,recall,fmeasure,fbeta_score])
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    # weights_path = COCO_WEIGHTS_PATH
    # if int(args.LOOP)>0:
    #     weights_path = '../../best.h5'
    # if not os.path.exists(args.LRPTS_DIR):
    #     os.mkdir(args.LRPTS_DIR)
    # tpc=os.path.join(args.LRPTS_DIR,'TS_of_loop'+str(args.LOOP))
    # tp=os.path.join(args.LRPTS_DIR,'TS_of_loop'+str(int(args.LOOP)-1)+'/')
    # if not os.path.exists(tpc):
    #     os.mkdir(tpc)
    # print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True)
    # else:
    # model.load_weights(weights_path, by_name=True)
    # model.summary()
    # model.uses_learning_phase=True
    #
    # model=modellib.remodel(model)
    # for layer in model.layers[:-3     z]:
    #     layer.trainable = False
    if args.command == "train":
        train_datagen = ImageDataGenerator(
            horizontal_flip=True)
        test_datagen = ImageDataGenerator()
        NP = r"/data3/dataL/LUNG_NEW_NP/"
        NP_remained = r"/data3/dataL/LUNG_NEW_NPTEST/"
        batchsize=4
        train_generator = train_datagen.flow_from_directory(
            NP,
            target_size=(224, 224),
            batch_size=batchsize)
        validation_generator = test_datagen.flow_from_directory(
            NP_remained,
            target_size=(224, 224),
            batch_size=batchsize)
        train_generator = SCALE3x3iterator(train_generator)
        validation_generator=SCALE3x3iterator(validation_generator)
    # train_generator = train_datagen.flow_from_directory(
    #     NP,
    #     target_size=(2000, 2000),
    #     batch_size=10)
    # train_generator=train_datagen.flow(alldata[:300],labs[:300],batch_size=10)
    # validation_generator = test_datagen.flow(alldata[300:],labs[300:],batch_size=10)
    #MaskrcnnPath = '../../logs/prcc/resnet_PRCC10.h5'

    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=4000,
    #     epochs=5,
    #     validation_data=validation_generator,
    #     validation_steps=10)
    # model.save('resnet_PRCC5.h5')

    # score = model.evaluate_generator(generator=validation_generator,
    #                                  workers=1,
    #                                  use_multiprocessing=False,
    #                                  verbose=0)
    #
    # print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
    # print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
    # print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
    # print('%s: %.2f%%' % (model.metrics_names[3], score[3] * 100))  # metrics1

    # names = [weight.name for layer in model.layers for weight in layer.weights]
    # weights = model.get_weights()
    # for name, weight in zip(names, weights):
    #     print(name, weight.shape)
    # MaskrcnnPath = '../../logs/prcc/resnet_PRCC0507vgg3.h5'
    # MaskrcnnPath = '../../logs/prcc/resnet_PRCC0508vgg1.h5'
    # model.load_weights(os.path.abspath(MaskrcnnPath), by_name=True)
    # MaskrcnnPath = 'resnet_lungNPresnet_C5_0.h5'
    # model.load_weights(MaskrcnnPath)
    if args.command == "train":
        for it in range(1,10):
            print('iter:::::::::{}:::::::::'.format(it))
            if parallel:
                cw={0:1,1:169/1.4,2:169/63,3:169/10,4:1}
                parallel_model.fit_generator(
                    train_generator,
                    steps_per_epoch=(train_generator.n/2)/(batchsize*2),
                    epochs=1,
                    class_weight=cw,
                    validation_data=validation_generator,
                    validation_steps=16)
                model.save('LUNGBIGDATA_{}.h5'.format(it))
                # # model.save(os.path.join(args.LRPTS_DIR,'classification_model_of_loop_'+str(args.LOOP)+'.h5'))
                # score = parallel_model.evaluate_generator(generator=validation_generator,
                #                                           workers=16,
                #                                           use_multiprocessing=True,
                #                                           verbose=0)
                #
                # print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
                # print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
                # print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
            else:
                model.fit_generator(
                    train_generator,
                    steps_per_epoch=0.2*(train_generator.n/2)/batchsize,
                    epochs=1,
                    validation_data=validation_generator,
                    validation_steps=16)
                model.save('LUNGBIGDATA_{}.h5'.format(it))
            # model.save(os.path.join(args.LRPTS_DIR,'classification_model_of_loop_'+str(args.LOOP)+'.h5'))
                score = model.evaluate_generator(generator=validation_generator,
                                                 workers=16,
                                                 use_multiprocessing=True,
                                                 verbose=0)

                print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
                print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
                print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
    if args.command == "detect":
        itern=6
        savp='TOSHOW_MULTISCALE+NORM_{}'.format(itern)
        if not os.path.exists(savp):
            os.mkdir(savp)
        MaskrcnnPath = 'LUNGBIGDATA_{}.h5'.format(itern)
        model.load_weights(os.path.abspath(MaskrcnnPath), by_name=True)
        DETECT_SET=False
        ONLY_SHOW_ORI=False
        if DETECT_SET:
            score = model.evaluate_generator(generator=validation_generator,
                                             workers=1,
                                             use_multiprocessing=False,
                                             verbose=0)

            print('%s: %.2f' % (model.metrics_names[0], score[0]))  # Loss
            print('%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))  # metrics1
            print('%s: %.2f%%' % (model.metrics_names[2], score[2] * 100))  # metrics2
        SVSDIR = r'/data2/山西省肿瘤30例_svs'  # '/data3/SCC58/incomp-20/'#/data1/wyj/M/datasets/data2/svs1/'#'/data1/wyj/M/datasets/data2/svs3conv/'
        # SVSDIR='/home/iftwo/data1/svs4conv/'
        TOTAL_CLASS_COUNTERS = np.zeros((5), dtype=np.int)
        id = -1
        USE_NORM = args.use_norm
            # import cv2
            # normalizer2.fit(cv2.imread(target))
        # LINE_IMGS=[]
        # imori=io.imread("/data1/wyj/M/datasets/SHANXI_ORI/2022-04380-1.svs/2022-04380-1.svs_i99_j61.jpg")
        # imori=io.imread("/data1/wyj/M/datasets/SHANXI_ORI/2022-04380-15.svs/2022-04380-15.svs_i61_j78.jpg")
        # imstain = io.imread("/data1/wyj/M/datasets/SHANXI_NORM/2022-04380-1.svs/2022-04380-1.svs_i99_j61.jpg")
        # # target = "/home/iftwo/1031733-9_x20384_y9408_.png"
        # imtarget=io.imread(target)
        # blankt = np.zeros((672, 672, 3), dtype=np.uint8)
        # blank=np.zeros((672,672,3),dtype=np.uint8)
        # LINE_IMGS.append(imori)
        # imori=normalizer2.transform(imori)
        # for channel in range(3):
        #     factor1=np.mean(imtarget[:,:,channel])/np.mean(imori[:,:,channel])
        #     blankt[:, :, channel]=imori[:,:,channel]*factor1
        # # factor1=np.mean(imtarget)/np.mean(imori)
        # # blankt=imori*factor1
        # # LINE_IMGS.append(blankt)
        # LINE_IMGS.append(normalizer2.transform(imori))
        # # LINE_IMGS.append(io.imread('~/trans2.jpg'))
        # for i in range(3):
        #     for j in range(3):
        #         blank[i*224:i*224+224,j*224:j*224+224,:]=imtarget[:,:,:3]
        # LINE_IMGS.append(blank)
        # predr = model.predict(np.array(LINE_IMGS))
        # plt.subplot(3,1,1)
        # plt.imshow(LINE_IMGS[0])
        # plt.subplot(3,1,2)
        # plt.imshow(LINE_IMGS[1])
        # plt.subplot(3,1,3)
        # plt.imshow(LINE_IMGS[2])
        # plt.show()
        for svsname in os.listdir(SVSDIR):
            id += 1
            if id < int(args.start):
                continue
            # if '1069555-28' not in svsname :#and '1069555-8' not in svsname:
            #     continue
            # if '2022-07661-6.svs' in svsname or '2022-08200-6.svs'  in svsname or '2022-04380-24.svs' in svsname or '2022-04380-1.svs' in svsname or '2022-04380-15.svs' in svsname:  # and not os.path.exists('TOSHOW2/{}_whole_classify.png'.format(svsname[:-4])):
            if '.svs' in svsname:
                SVSPATH = SVSDIR + '/' + svsname
                print(SVSPATH + '   id:{}'.format(id))
                try:
                    slide = openslide.open_slide(SVSPATH)
                except:
                    print('cant open it!!!!!!!!!!!!!!!!!')
                    continue
                if USE_NORM:
                    from Fast_WSI_Color_Norm.Run_ColorNorm import run_colornorm
                    target = "/data3/dataL/LUNG_NEW_NP/2/1021724-82022-07-01_22_33_43_x30688_y29120_.png"  # "/home/iftwo/1031733-9_x20384_y9408_.png"
                    target='/data2/山西省肿瘤30例_svs/2022-04380-16.svs'# nothing for 07661-6
                    # target='/data2/山西省肿瘤30例_svs/2022-09575-1.svs' #"/data3/dataL/LUNG_NEW_NP/2/1041047-8_x37408_y21504_.png"
                    background_correction = True
                    nstains = 2  # number of stains
                    lamb = 0.01  # default value sparsity regularization parameter
                    output_direc = "/data3/kfb_norm/"
                    ORI_CUDA_GPUS=os.environ['CUDA_VISIBLE_DEVICES']
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # use only CPU
                    # tf.config.set_visible_devices(device_type='CPU')
                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
                    config = tf.ConfigProto(device_count={'GPU': 0},log_device_placement=False,gpu_options=gpu_options)
                    # config = tf.ConfigProto(log_device_placement=False, gpu_options=None)
                    NORMED_SVSPATH = run_colornorm(SVSPATH, target, nstains, lamb, output_direc, level=0,
                                  background_correction=background_correction, config=config)
                    slide = openslide.open_slide(NORMED_SVSPATH)
                    os.environ['CUDA_VISIBLE_DEVICES'] = ORI_CUDA_GPUS
                    # gpus = tf.config.list_physical_devices(device_type='GPU')
                    # tf.config.set_visible_devices(devices=gpus[0:2], device_type='GPU')
                [w, h] = slide.level_dimensions[0]
                downscale = h / 2000
                data_gen = DeepZoomGenerator(slide, tile_size=224, overlap=224, limit_bounds=False)
                # print('生成的层数:', data_gen.level_count)
                # print('切分成的块数:', data_gen.tile_count)
                # print('每层尺寸大小:', data_gen.level_dimensions)
                # print('切分的每层的块数:', data_gen.level_tiles)
                # print('w:', str(w), '  h:', str(h), '  downscale: {}'.format(downscale))
                nailmap = slide.get_thumbnail((200000000, 2000))
                plt.imshow(nailmap)
                if USE_NORM:
                    plt.show()
                    plt.imshow(nailmap)
                if ONLY_SHOW_ORI:
                    plt.savefig('TOSHOW9_GT/{}.png'.format(svsname[:-4]))
                    continue
                num_w = int(np.floor(w / 224))
                num_h = int(np.floor(h / 224))
                SWITCH = ['c', 'g', 'r', 'b', 'white', ]
                TEMP_CLASS_COUNTERS = np.zeros((5), dtype=np.int)
                BAD_PATCH = 0
                NEED_SAVE_NORM=False
                NEED_SAVE_ORI=False
                ALREADY_SAVED=False
                for i in range(1,num_w-1):
                    LINE_IMGS = []
                    for j in range(1,num_h-1):
                        if ALREADY_SAVED:
                            try:
                                img = io.imread(
                                '/data1/wyj/M/datasets/SHANXI_NORM/{}/{}_i{}_j{}.jpg'.format(svsname, svsname, i, j))
                            except:
                                img=io.imread('/data1/wyj/M/datasets/SHANXI_ORI/{}/{}_i{}_j{}.jpg'.format(svsname, svsname, i, j))
                        else:
                            try:
                                img = np.array(data_gen.get_tile(data_gen.level_count - 1, (i, j)))
                                if NEED_SAVE_ORI:
                                    if not os.path.exists('/data1/wyj/M/datasets/SHANXI_ORI/{}'.format(svsname)):
                                        os.mkdir('/data1/wyj/M/datasets/SHANXI_ORI/{}'.format(svsname))
                                    io.imsave(
                                        '/data1/wyj/M/datasets/SHANXI_ORI/{}/{}_i{}_j{}.jpg'.format(svsname, svsname, i, j),
                                        img)
                                # if USE_NORM:
                                #     img = normalizer2.transform(img)
                                #     if NEED_SAVE_NORM:
                                #         if not os.path.exists('/data1/wyj/M/datasets/SHANXI_NORM/{}'.format(svsname)):
                                #             os.mkdir('/data1/wyj/M/datasets/SHANXI_NORM/{}'.format(svsname))
                                #         io.imsave('/data1/wyj/M/datasets/SHANXI_NORM/{}/{}_i{}_j{}.jpg'.format(svsname,svsname,i,j),img)
                            except:
                                img = np.zeros((224*3, 224*3, 3), dtype=np.int)
                                BAD_PATCH += 1
                        LINE_IMGS.append(img)
                    # if USE_NORM:
                    #     continue
                    if BAD_PATCH == 0:
                        # predr = parallel_model.predict(np.array(LINE_IMGS))
                        predr = model.predict(np.array(LINE_IMGS))
                        # for j in range(num_h):
                        #     pred_CU = predr[j, :]
                        #     if np.max(pred_CU) == pred_CU[3] and pred_CU[3] < 0.8:
                        #         pred_CU[3] = 0
                        #     if np.max(pred_CU) == pred_CU[2] and pred_CU[2] < 0.7:
                        #         if pred_CU[2] + pred_CU[3] < 0.9:
                        #             pred_CU[4] += pred_CU[2]
                        #             pred_CU[2] = 0
                        #         else:
                        #             pred_CU[4] += 1
                        pred_class = np.argmax(predr, axis=1)
                        PRED_CLASS_H_SIZE=np.zeros((num_h,),dtype=np.uint8)
                        PRED_CLASS_H_SIZE[1:num_h-1]=pred_class
                        pred_class=PRED_CLASS_H_SIZE
                        for j in range(1,num_h-1):
                            TOTAL_CLASS_COUNTERS[pred_class[j]] += 1
                            TEMP_CLASS_COUNTERS[pred_class[j]] += 1
                            plt.gca().add_patch(plt.Rectangle(
                                xy=(i * (224) / downscale , j * (224) / downscale ) ,
                                width=(224) / downscale,
                                height=(224) / downscale,
                                edgecolor=SWITCH[pred_class[j]],
                                fill=False, linewidth=1))
                    # plt.show()
                    # plt.close()
                    # plt.gca().add_patch(plt.Rectangle(
                    #     xy=(100, 100),
                    #     width=(224),
                    #     height=(224),
                    #     edgecolor='gold',
                    #     fill=False, linewidth=5))
                if BAD_PATCH == 0:
                    g = TEMP_CLASS_COUNTERS[1]
                    r = TEMP_CLASS_COUNTERS[2]
                    b = TEMP_CLASS_COUNTERS[3]
                    try:
                        mpr = b / (r + g + b)
                    except:
                        mpr = 0
                    plt.title("mpr={} , detail : {} red ; {} blue ; {} green ".format(mpr, r, b, g), y=-0.15, fontsize=10)
                    plt.savefig('{}/{}_whole_classify_mpr{}.png'.format(savp, svsname[:-4], mpr))
                    # plt.show()
                    print("current image , blue / ( red + blue + green ) = {}".format(mpr))
                    print("detail : {} red ; {} blue ; {} green ".format(r, b, g))
                else:
                    print('BAD PATCH:{}'.format(BAD_PATCH))
                if args.plt:
                    plt.show()
                plt.close()
                # orig_imgs=[]


        # target=r'/data1/wyj/M/datasets/LUNG_OD_4_CLASS_TESTSET/3'
        # oser=os.listdir(target)
        # oser.sort()
        # # oser=oser[-60:-30]
        # for fname in oser:
        #     print(target+"/{}".format(fname))
        #     orig_imgs.append(img_to_array(load_img(target+"/{}".format(fname))))
        # predr=model.predict(np.array(orig_imgs))
        # for i in range(len(oser)):
        #     ks = print(np.argmax(predr[i, :]))

        # k=[y2lab(predr[i])
        #    for i in range(predr.shape[0]) if print(oser[i])==None]
        # orig_imgs = [img_to_array(load_img("/data1/wyj/M/datasets/dataset-master/JPEGImages/{}".format(fname))) for fname in os.listdir("/data1/wyj/M/datasets/dataset-master/JPEGImages/")[-20:]]
        # predr=model.predict(np.array(orig_imgs))
        # print(predr)
        # k=[y2lab(predr[i]) for i in range(predr.shape[0])]
    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)
    #
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # NP = r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\samples\nucleus\MyNP'
    # train_generator = train_datagen.flow_from_directory(
    #     NP,
    #     target_size=(224, 224),
    #     batch_size=20)
    #
    # validation_generator = test_datagen.flow_from_directory(
    #     NP,
    #     target_size=(224, 224),
    #     batch_size=20)
    #
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=30000,
    #     epochs=1,
    #     validation_data=validation_generator,
    #     validation_steps=100)
    # model.save('vgg6w.h5')


    # Train or evaluate
    # if args.command == "train":
    #     train(model, args.dataset, args.subset,iter=int(args.iteration))
    # elif args.command == "detect":
    #     detect(model, args.dataset, args.subset,iter=int(args.iteration))
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'detect'".format(args.command))
