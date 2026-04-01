import matplotlib
# Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib
import shutil,os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from xml.dom.minidom import parse,parseString
from shapely.geometry import Polygon
import copy
import os
# openslide-bin-path为 openslide 的bin文件夹绝对路径。

import openslide,numpy
from openslide.deepzoom import DeepZoomGenerator
target=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\samples\nucleus\shit'
sourcedir=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\images'
dd1=r'D:\BaiduNetdiskDownload\BrestCancer\Mask_RCNN\datasets\nucleus - 副本\stage1_train'
dd=r'../../datasets/MoNuSACGT\\stage1_train\\stage1_train'
import os
import sys
import random
import re
import time
from skimage import io
import numpy as np
from matplotlib.path import Path
import warnings
warnings.filterwarnings('error',category=UserWarning)
def get_centerpoint(lis):
    area = 0.0
    x,y = 0.0,0.0

    a = len(lis)
    for i in range(a):
        lat = lis[i][0] #weidu
        lng = lis[i][1] #jingdu

        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]

        else:
            lat1 = lis[i-1][0]
            lng1 = lis[i-1][1]

        fg = (lat*lng1 - lng*lat1)/2.0

        area += fg
        x += fg*(lat+lat1)/3.0
        y += fg*(lng+lng1)/3.0

    x = x/area
    y = y/area

    return x,y
# 4278255615 12青色
# 4278255360  曲线11 绿色
# 4294901760 曲线10 红色
# 4278190335  1蓝色
SWITCH={
    '4278255615' : [0,'c','Normal_cell'],
    '4278255360' : [1,'g','Nacrotic'],
    '4294901760' : [2,'r','Tumor_Bed'],
    '4278190335' : [3,'b','Remained_Cancer_Cell'],
}
SWITCH_C={
    '4278255615' : [0,'c','Normal_cell'],
    '4278255360' : [1,'g','Nacrotic'],
    '4294901760' : [2,'r','Tumor_Bed'],
    '4278190335' : [3,'b','Remained_Cancer_Cell'],
    'back' : [4,'white','Remained_Cancer_Cell'],
}
SAVEABLE=False
TO_SHOW=True
SKIP=False
def union_of_polygons(ALL_POLYGON):
    UNION_POLY=Polygon([(0.0001,0.0001),(0.0002,0.0001),(0.0002,0.0002),(0.0001,0.0002)])
    for pid in range(len(ALL_POLYGON)):
        UNION_POLY = UNION_POLY.union(ALL_POLYGON[pid])
    return UNION_POLY
def label_of_this_polygon(this_p,ps):
    min_class_area=None
    prefer_id = 4
    for id,classp in enumerate(ps):
        intersact_rate_with_C = this_polygon.intersection(classp).area / this_polygon.area
        if intersact_rate_with_C>0.6:
            if min_class_area is None:
                min_class_area=classp.area
                prefer_id = id
            else:
                if classp.area<min_class_area:
                    prefer_id=id
                    min_class_area=classp.area
    return prefer_id
if __name__ == '__main__':
    src_folder_name = r'/data3/kfb/'  # KFB文件所在文件夹
    src=r'svs'
    SVSPATH= '/data3/kfb_svs/{}.svs'
    bis = 0
    NP='/data3/dataL/LUNG_NEW_NP'
    NPTEST='/data3/dataL/LUNG_NEW_NPTEST'
    train_or_test = [NP+'/{}/', NPTEST+'/{}/']
    if not os.path.exists(NP):
        os.mkdir(NP)
    for i in range(4+1):
        target=train_or_test[0].format(i)
        if not os.path.exists(target):
            os.mkdir(target)
    if not os.path.exists(NPTEST):
        os.mkdir(NPTEST)
    for i in range(4+1):
        target=train_or_test[1].format(i)
        if not os.path.exists(target):
            os.mkdir(target)
    EXIST_COLORS=[]
    ACCEPTED_CLASS_VOLUME=np.zeros((5),dtype=np.int)
    WSI_ID = -1
    for xml in os.listdir(src_folder_name):
        if xml.endswith('.kfb.Ano'):
            WSI_ID += 1
            # if WSI_ID<=297 :
            #     continue
            ALL_POLYGON_OF_SWITCH = []
            for C in SWITCH.keys():
                TEMP_CLASS = []
                ALL_POLYGON_OF_SWITCH.append(TEMP_CLASS)
            ALL_POLYGON = []
            ALL_POLYGON_tumor = []
            ALL_POLYGON_nacro = []
            ALL_POLYGON_normal = []
            im_no = 0
            path=src_folder_name+'/'+xml#+'\828919-62022-04-25_16_25_04.kfb.Ano'
            slidepath = SVSPATH.format(xml[:-8])
            file_object = open(path)
            ori_xml = file_object.read()
            file_object.close()
            pro_xml = ori_xml.replace("utf-8", "gb2312")
            Tree = parseString(pro_xml)
            # path='/data1/wyj/M/datasets/ccRCC_TCGA/data/'+file+'/'+xml
            # Tree=parse(path)
            root=Tree.documentElement
            # vs = root.getElementsByTagName('Vertex')
            regs = root.getElementsByTagName('Regions')
            reg = regs[0].getElementsByTagName('Region')
            try:
                slide = openslide.open_slide(slidepath)
            except:
                print('bad wsi!!!!!!!!!!!!!!!!')
                continue
            [w, h] = slide.level_dimensions[0]
            downscale = h / 2000
            # data_gen = DeepZoomGenerator(slide, tile_size=50, overlap=0, limit_bounds=False)
            print('{}   id:{}'.format(slidepath,WSI_ID))
            # print('生成的层数:', data_gen.level_count)
            # print('切分成的块数:', data_gen.tile_count)
            # print('每层尺寸大小:', data_gen.level_dimensions)
            # print('切分的每层的块数:', data_gen.level_tiles)
            # print('w:', str(w), '  h:', str(h), '  downscale: {}'.format(downscale))
            nailmap = slide.get_thumbnail((20000000, 2000))
            plt.imshow(nailmap)
            for regid in range(len(reg)):#maybe 4 : lanse hongse lvse qingse
                COLOR=reg[regid].getAttribute('Color')
                # print('Color:{}'.format(COLOR))
                vs=reg[regid].getElementsByTagName('Vertices')
                v=vs[0].getElementsByTagName('Vertice')
                vx_array=[]
                for vertice in v:
                    try:
                        vx_array.append([float(vertice.getAttribute('X'))*20,float(vertice.getAttribute('Y'))*20])
                    except:
                        pass
                vx_array=np.array(vx_array)
                try:
                    CONTOUR = Path(vx_array)
                except:
                    print('fail xml')
                    continue
                bbox = np.array(CONTOUR.get_extents(), dtype=float)
                x1, y1, x2, y2 = bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1]
                vx_array_x = vx_array[:, 0] - x1
                vx_array_y = vx_array[:, 1] - y1
                try:
                    poly_contour = Polygon(vx_array - [x1, y1] + [bis, bis]).buffer(0.001)
                except:
                    continue
                if '4294' in COLOR:
                    COLOR='4294901760'
                if COLOR in list(SWITCH.keys()):
                    TEMP_polygom = Polygon(vx_array).buffer(0.001)
                    ALL_POLYGON_OF_SWITCH[SWITCH[COLOR][0]].append(TEMP_polygom)
                    plotx = vx_array[:, 0] / downscale
                    ploty = vx_array[:, 1] / downscale
                    plt.plot(plotx, ploty, SWITCH[COLOR][1], linewidth=0.5)

                    # print('Color:{}'.format(COLOR))
                    pass
                else:
                    if COLOR not in EXIST_COLORS:
                        EXIST_COLORS.append(COLOR)
                        print('COLORS NEW: {}'.format(EXIST_COLORS))
            # plt.savefig('TOSHOW4/{}.png'.format(xml[:-8]))

            # plt.show()
            UNION_POLYGON_OF_SWITCH = []
            for polyset in ALL_POLYGON_OF_SWITCH:
                UNION_POLY = union_of_polygons(polyset)
                UNION_POLYGON_OF_SWITCH.append(UNION_POLY)
            green_area=UNION_POLYGON_OF_SWITCH[1].area
            red_area = UNION_POLYGON_OF_SWITCH[2].area
            blue_area = UNION_POLYGON_OF_SWITCH[3].area
            mpr= blue_area / red_area
            mpr=round(mpr,3)
            print('mpr={} blue={} red={} green={}'.format(mpr,blue_area,red_area,green_area))
            xn = w // 224
            # print('XN' + str(xn))
            yn = h // 224
            # print('YN' + str(yn))
            ACCEPTED_IM_NUMS = np.zeros((5), dtype=np.int)
            if not SKIP:
                for x_ in range(int(xn)):
                    for y_ in range(int(yn)):
                        # if np.sum(ACCEPTED_IM_NUMS) >= 500 * (TEMP_EXIST_COLORS + 1):
                        #     break
                        this_polygon = Polygon(
                            [(x_ * 224 + bis, y_ * 224 + bis), (x_ * 224 + 224 + bis, y_ * 224 + bis),
                             (x_ * 224 + 224 + bis, y_ * 224 + 224 + bis), (x_ * 224 + bis, y_ * 224 + 224 + bis)])
                        if SAVEABLE:
                            # savp=neg+name[:-4]+'_{}_{}.png'.format(str(arg1),str(arg2))
                            # intersact_rate_with_BED = this_polygon.intersection(UNION_POLY).area / this_polygon.area
                            # intersact_rate_with_tumor = this_polygon.intersection(
                            #     UNION_POLY_tumor).area / this_polygon.area
                            # intersact_rate_with_nacro = this_polygon.intersection(
                            #     UNION_POLY_nacro).area / this_polygon.area
                            # intersact_rate_with_normal = this_polygon.intersection(
                            #     UNION_POLY_normal).area / this_polygon.area
                            LABEL_of_P = label_of_this_polygon(this_polygon, UNION_POLYGON_OF_SWITCH)
                            if ACCEPTED_IM_NUMS[LABEL_of_P] < 10000:
                                savp = train_or_test[0].format(LABEL_of_P) + xml[
                                                                                     :-8] + '_x{}_y{}_{}.png'.format(
                                    int(x_ * 224), int(y_ * 224), im_no)
                                try:
                                    tile = numpy.array(
                                        slide.read_region(((x_ * 224), int(y_ * 224)), 0, (224, 224)))
                                except:
                                    break
                                try:
                                    io.imsave(savp, tile)
                                except:
                                    continue
                                ACCEPTED_IM_NUMS[LABEL_of_P] += 1
                                ACCEPTED_CLASS_VOLUME[LABEL_of_P] += 1
                                im_no = im_no + 1
                                if TO_SHOW:
                                    plt.gca().add_patch(plt.Rectangle(
                                        xy=(x_ * (224) / downscale, y_ * (224) / downscale),
                                        width=(224) / downscale,
                                        height=(224) / downscale,
                                        edgecolor=SWITCH_C[list(SWITCH_C.keys())[LABEL_of_P]][1],
                                        fill=False, linewidth=1))
                    else:
                        continue
                    break
            if TO_SHOW:
                savpath='TOSHOW5/{}_mpr{}.png'.format(xml[:-8],mpr)
                if not os.path.exists(savpath):
                    plt.savefig(savpath)
                # plt.show()
            plt.close()
                # try:
                #     if COLOR.endswith('4294901760'):
                #         #
                #         plt.plot(vx_array[:,0],vx_array[:,1],SWITCH[COLOR][1],linewidth=0.5)
                #         continue
                #     if COLOR.endswith('4278255360'):#nacro
                #         #print('Color:{}'.format(COLOR))
                #         plt.plot(vx_array[:,0],vx_array[:,1],SWITCH[COLOR][1],linewidth=0.5)
                #         continue
                #     if COLOR.endswith('4278190335'):#tumor
                #         #print('Color:{}'.format(COLOR))
                #         plt.plot(vx_array[:,0],vx_array[:,1],SWITCH[COLOR][1],linewidth=0.5)
                #         continue
                #     if COLOR.endswith('4278255615'):
                #         #print('Color:{}'.format(COLOR))
                #         plt.plot(vx_array[:,0],vx_array[:,1],SWITCH[COLOR][1],linewidth=0.5)
                #         continue
                # except:
                #     pass

            # print('vx_array:{}'.format(vx_array))
            # for reg in range(2):
            #     vs =ANN[reg].getElementsByTagName('Vertex')
            #     v=vs[1]
            #     picpath=path.replace('.xml','.svs')
            #     print('running no:{}'.format(xml))
            #     slide = openslide.open_slide(picpath)
            #     [w, h] = slide.level_dimensions[0]
            #     downscale = h / 2000
            #     data_gen = DeepZoomGenerator(slide, tile_size=50, overlap=0, limit_bounds=False)
            #     print('生成的层数:', data_gen.level_count)
            #     print('切分成的块数:', data_gen.tile_count)
            #     print('每层尺寸大小:', data_gen.level_dimensions)
            #     print('切分的每层的块数:', data_gen.level_tiles)
            #     print('w:', str(w), '  h:', str(h), '  downscale: {}'.format(downscale))
            #     nailmap = slide.get_thumbnail((20000000, 2000))
                # for i in range(len(vs)//4):
                #     v1=vs[i*4]
                #     v2=vs[i*4+1]
                #     v3=vs[i*4+2]
                #     x1=v1.getAttribute('X')
                #     x2=v2.getAttribute('X')
                #     y1=v2.getAttribute('Y')
                #     y2=v3.getAttribute('Y')
                #     print(NP,x1,x2,y1,y2)
                #     x1,y1,x2,y2=int(float(x1)),int(float(y1)),int(float(x2)),int(float(y2))
                #     if x1>x2:
                #         k=x1
                #         x1=x2
                #         x2=k
                #     if y1>y2:
                #         k=y1
                #         y1=y2
                #         y2=k
                #     args=[x1,y1,x2,y2]
                #     xn = ((args[2] - args[0]) ) // 224
                #     print('XN' + str(xn))
                #     yn = ((args[3] - args[1]) ) // 224
                #     print('XN' + str(yn))
                #     bis = 0
                #     tilemaps = slide.read_region((int((args[0] ) - bis), int((args[1]) - bis)), 0,
                #                                  (224 * xn, 224 * yn))
                #     plt.imshow(tilemaps)
                #     plt.gca().add_patch(plt.Rectangle(
                #         xy=(bis, bis),
                #         width=((args[2] - args[0]) ),
                #         height=((args[3] - args[1]) ),
                #         edgecolor=[0, 0, 1],
                #         fill=False, linewidth=1))
                #     for x_ in range(int(xn)):
                #         for y_ in range(int(yn)):
                #             plt.gca().add_patch(plt.Rectangle(
                #                 xy=((x_ * 224) + bis, (y_ * 224) + bis),
                #                 width=224,
                #                 height=224,
                #                 edgecolor=[0, 0, 1],
                #                 fill=False, linewidth=1))
                #     # plt.show()
                #     for x_ in range(int(xn)):
                #         for y_ in range(int(yn)):
                #             tile = numpy.array(
                #                 slide.read_region((int(args[0]  + x_ * 224), int(args[1]  + y_ * 224)), 0,
                #                                   (224, 224)))
                #             # savp=neg+name[:-4]+'_{}_{}.png'.format(str(arg1),str(arg2))
                #             npphase='positive' if NP==0 else 'negative'
                #             savp = '/data1/wyj/M/datasets/ccrccNP/{}/'.format(npphase) + xml[:-4] + '_{}.png'.format(im_no)
                #             im_no = im_no + 1
                #             io.imsave(savp, tile)









# for file in os.listdir('')
    #
    # k=np.array([250],dtype=np.uint8)
    # f=k+k
    # print(f)
    # from scipy.io import loadmat
    # mat=loadmat("/data1/wyj/M/datasets/ccrcc/Test/Labels/high_grade_ccrcc_10.mat")
    # for i in range(1,96):
    #     ins=mat['instance_map']
    #     insmask=ins==i
    #     if np.max(insmask)==False:
    #         print('fuck')
    # dirp='/data1/wyj/M/datasets/ccRCC_TCGA/data/'
    # neg = 'negative/'
    # pos = 'positive/'
    # nail = 'thumbnail/'
    # names=[r'TCGA-BP-4159-01Z-00-DX1',
    #
    #        ]
    #
    # for fname in os.listdir(dirp):
    #     fnamed=os.listdir(dirp+'/'+fname)[0]
    #     path =dirp+'/'+fname+'/'+fnamed
    #     print('running no:{}'.format(path))
    #     slide = openslide.open_slide(path)
    #     [w, h] = slide.level_dimensions[0]
    #     downscale=h/2000
    #     data_gen = DeepZoomGenerator(slide, tile_size=50, overlap=0, limit_bounds=False)
    #     print('生成的层数:', data_gen.level_count)
    #     print('切分成的块数:', data_gen.tile_count)
    #     print('每层尺寸大小:', data_gen.level_dimensions)
    #     print('切分的每层的块数:', data_gen.level_tiles)
    #     print('w:', str(w),'  h:',str(h),'  downscale: {}'.format(downscale))
    #     # nailmap = slide.get_thumbnail((20000000, 2000))
    #     nailmap = slide.get_thumbnail((w/100, h/100))
    #     nailmap.save('/data1/wyj/M/datasets/ccRCC_TCGA/crop/'+fnamed[:-4]+'.tiff')