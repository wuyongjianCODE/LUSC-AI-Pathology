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
    '4278255615' : [1,'cyan','Normal_cell'],
    '4278255360' : [2,'lightgreen','Nacrotic'],
    '4294901760' : [3,'red','Tumor_Bed'],
    '4278190335' : [4,'blue','Remained_Cancer_Cell'],
}
SAVEABLE=True
def union_of_polygons(ALL_POLYGON):
    UNION_POLY=Polygon([(0.0001,0.0001),(0.0002,0.0001),(0.0002,0.0002),(0.0001,0.0002)])
    for pid in range(len(ALL_POLYGON)):
        UNION_POLY = UNION_POLY.union(ALL_POLYGON[pid])
    return UNION_POLY
def label_of_this_polygon(BED,tumor,nacro,normal):
    if tumor>0.6:
        return 2
    if nacro>0.6:
        return 3
    if normal>0.6:
        return 4
    if BED>0.6:
        return 1
    return 0
if __name__ == '__main__':
    src_folder_name = r'/data1/wyj/M/datasets/data2/svs3conv/'  # KFB文件所在文件夹
    src=r'svs'
    SVSPATH= '/data1/wyj/M/datasets/data2/svs3conv/{}.svs'
    bis = 224
    train_or_test = ['/data1/wyj/M/datasets/LUNG_NP/{}/', '/data1/wyj/M/datasets/LUNG_NPTEST/{}/']
    if not os.path.exists('/data1/wyj/M/datasets/LUNG_NP/'):
        os.mkdir('/data1/wyj/M/datasets/LUNG_NP/')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NP/Negative')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NP/Positive')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NP/Positive_1')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NP/Positive_2')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NP/Positive_3')
    if not os.path.exists('/data1/wyj/M/datasets/LUNG_NPTEST/'):
        os.mkdir('/data1/wyj/M/datasets/LUNG_NPTEST/')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NPTEST/Negative')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NPTEST/Positive')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NPTEST/Positive_1')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NPTEST/Positive_2')
        os.mkdir('/data1/wyj/M/datasets/LUNG_NPTEST/Positive_3')
    for xml in os.listdir(src_folder_name):
        if xml.endswith('.kfb.Ano'):
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
            slide = openslide.open_slide(slidepath)
            [w, h] = slide.level_dimensions[0]
            downscale = h / 2000
            data_gen = DeepZoomGenerator(slide, tile_size=50, overlap=0, limit_bounds=False)
            print(slidepath)
            print('生成的层数:', data_gen.level_count)
            print('切分成的块数:', data_gen.tile_count)
            print('每层尺寸大小:', data_gen.level_dimensions)
            print('切分的每层的块数:', data_gen.level_tiles)
            print('w:', str(w), '  h:', str(h), '  downscale: {}'.format(downscale))
            nailmap = slide.get_thumbnail((20000000, 2000))
            for regid in range(len(reg)):#maybe 4 : lanse hongse lvse qingse
                COLOR=reg[regid].getAttribute('Color')
                # print('Color:{}'.format(COLOR))
                vs=reg[regid].getElementsByTagName('Vertices')
                v=vs[0].getElementsByTagName('Vertice')
                vx_array=[]
                for vertice in v:
                    vx_array.append([float(vertice.getAttribute('X'))*20,float(vertice.getAttribute('Y'))*20])
                vx_array=np.array(vx_array)
                CONTOUR=Path(vx_array)
                bbox=np.array(CONTOUR.get_extents(),dtype=float)
                x1,y1,x2,y2=bbox[0,0],bbox[0,1],bbox[1,0],bbox[1,1]
                vx_array_x=vx_array[:,0]-x1
                vx_array_y=vx_array[:,1]-y1
                poly_contour = Polygon(vx_array - [x1, y1] + [bis, bis]).buffer(0.001)
                if COLOR.endswith('4294901760'):
                    TEMP_polygom=Polygon(vx_array).buffer(0.001)
                    ALL_POLYGON.append(TEMP_polygom)
                    print('Color:{}'.format(COLOR))
                    continue
                if COLOR.endswith('4278255360'):#nacro
                    TEMP_polygom=Polygon(vx_array).buffer(0.001)
                    ALL_POLYGON_nacro.append(TEMP_polygom)
                    print('Color:{}'.format(COLOR))
                    continue
                if COLOR.endswith('4278190335'):#tumor
                    TEMP_polygom=Polygon(vx_array).buffer(0.001)
                    ALL_POLYGON_tumor.append(TEMP_polygom)
                    print('Color:{}'.format(COLOR))
                    continue
                if COLOR.endswith('4278255615'):
                    TEMP_polygom=Polygon(vx_array).buffer(0.001)
                    ALL_POLYGON_normal.append(TEMP_polygom)
                    print('Color:{}'.format(COLOR))
                    continue

            UNION_POLY=union_of_polygons(ALL_POLYGON) #以下这四个变量 是由多个｛医生标注的多边形｝组合而成的复合多边形；例如，UNION_POLY_tumor能表示全体肿瘤区域，是一个复合多边形，而ALL_POLYGON_tumor只是存储了多个多边形的list
            UNION_POLY_tumor = union_of_polygons(ALL_POLYGON_tumor)
            UNION_POLY_nacro = union_of_polygons(ALL_POLYGON_nacro)
            UNION_POLY_normal = union_of_polygons(ALL_POLYGON_normal)

            xn = w // 224
            # print('XN' + str(xn))
            yn = h// 224
            # print('YN' + str(yn))
            for x_ in range(int(xn)):
                for y_ in range(int(yn)):
                    this_polygon=Polygon([(x_*224+bis,y_*224+bis),(x_*224+224+bis,y_*224+bis),(x_*224+224+bis,y_*224+224+bis),(x_*224+bis,y_*224+224+bis)])
                    if SAVEABLE:
                        tile = numpy.array(
                            slide.read_region(((x_ * 224), int(y_ * 224)), 0,(224, 224)))
                        # savp=neg+name[:-4]+'_{}_{}.png'.format(str(arg1),str(arg2))
                        intersact_rate_with_BED = this_polygon.intersection(UNION_POLY).area / this_polygon.area#这里计算｛代表某个类全体的复合多边形｝和｛当前这个224*224patch视作的多边形｝的重合率
                        intersact_rate_with_tumor = this_polygon.intersection(UNION_POLY_tumor).area / this_polygon.area
                        intersact_rate_with_nacro = this_polygon.intersection(UNION_POLY_nacro).area / this_polygon.area
                        intersact_rate_with_normal = this_polygon.intersection(UNION_POLY_normal).area / this_polygon.area
                        LABEL_of_P=label_of_this_polygon(intersact_rate_with_BED,intersact_rate_with_tumor,intersact_rate_with_nacro,intersact_rate_with_normal)
                        subdirs=['Negative','Positive','Positive_1','Positive_2','Positive_3']
                        savp = train_or_test[im_no % 2].format(subdirs[LABEL_of_P]) + xml[:-8] + '_x{}_y{}_{}.png'.format(int(x_ * 224), int(y_ * 224), im_no)
                        im_no = im_no + 1
                        try:
                            io.imsave(savp, tile)
                        except:
                            pass

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