import matplotlib
matplotlib.use('Agg')
import shutil,os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from xml.dom.minidom import parse,parseString
from shapely.geometry import Polygon
import copy
import os
import imageio
# openslide-bin-path为 openslide 的bin文件夹绝对路径。

import openslide,numpy
from openslide.deepzoom import DeepZoomGenerator
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

TO_SHOW =False
SAVEABLE=True
TO_SHOW_PLT=False

DICT = {
    '16711808': '0',    #品红
    '8454143': '1',    #淡蓝
    '255': '2',    #纯蓝
    '16711680': '3',  #纯红
    '65280': '4',   #纯绿
    '0': '5',    #黑

    '4194368': '6',
    '4259584': '6',
    '8454016': '6',
    '4259584': '6',
    '65535': '6',
    '8453888': '6',
    '12632256': '6',
    '8388863': '6',
}

#合并多个多边形并返回一个包裹全部多边形的大的
def union_of_polygons(ALL_POLYGON):
    UNION_POLY=Polygon([(0.0001,0.0001),(0.0002,0.0001),(0.0002,0.0002),(0.0001,0.0002)])
    for pid in range(len(ALL_POLYGON)):
        UNION_POLY = UNION_POLY.union(ALL_POLYGON[pid])
    return UNION_POLY


#ps：UNION_POLYGON_OF_SWITCH，单张svs中的所有多边形集合
#this_p: 切出的tile图像
def label_of_this_polygon(this_p,ps):
    min_class_area=None
    prefer_id = 6
    for id,classp in enumerate(ps):
        intersect_rate_with_C = this_p.intersection(classp).area / this_p.area
        if intersect_rate_with_C>0.6:
            if min_class_area is None:
                min_class_area=classp.area
                prefer_id = id
            else:
                if classp.area<min_class_area:
                    prefer_id=id
                    min_class_area=classp.area
    return prefer_id


if __name__ == '__main__':
    #这三个路径分别是（r表示原始字符串，不受转义字符影响）
    src_folder_name = r'/data3/pathology_label/'  # KFB DIR 文件所在文件夹
    plabel=r'/data3/special/'
    src=r'svs'
    #SVSPATH= '/data2/kfb_svs/{}.svs'
    bis = 0
    NP='/data3/datasets/KIDNEY_NP_BIGDATA_1/'
    NPTEST = '/data3/datasets/KIDNEY_NPTEST_BIGDATA_1/'
    train_or_test = [NP+'{}/', NPTEST+'{}/']
    #创建分类别的文件夹，名称为'NP/{0~6}/'
    if not os.path.exists(NP):
        os.mkdir(NP)
    ##for i in range(6+1):
    for i in range(7):
        target=NP+'{}/'.format(i)
        if not os.path.exists(target):
            os.mkdir(target)
    if not os.path.exists(NPTEST):
        os.mkdir(NPTEST)
    #for i in range(6+1):
    for i in range(7):
        target=NPTEST+'{}/'.format(i)
        if not os.path.exists(target):
            os.mkdir(target)
    
    WSI_ID = -1

    count = 0
    for xmldir in os.listdir(src_folder_name)[:]:
        count = count+1
        print(count)
        for xml in os.listdir(src_folder_name+xmldir):
            if xml.endswith('.svs') :
                if os.path.exists(plabel+xml[:-4]+'.xml'):   #检查.svs对应的.xml是否存在
                    WSI_ID += 1
                    path = plabel + xml[:-4] + '.xml'
                else:
                    continue
                if WSI_ID<686:
                    continue

                ALL_POLYGON_OF_SWITCH = []
                #im_no = 0
                for C in range(7):
                    TEMP_CLASS = []
                    ALL_POLYGON_OF_SWITCH.append(TEMP_CLASS)

                im_num = 0
                #+'\828919-62022-04-25_16_25_04.xml'plabel+xml[:-4]+'.xml'
                slidepath = src_folder_name + xmldir + '/' + xml      #xml实际上是.svs文件名
                file_object = open(path)
                ori_xml = file_object.read()
                file_object.close()
                pro_xml = ori_xml.replace("utf-8", "gb2312")    #pro_xml:经过编码替换后的.xml文件内容
                
                
                try:
                    Tree = parseString(pro_xml)
                except:
                    print('{} wrong!!!!!!!!!!!!!!!!!!!!!!'.format(xml))
                    continue
                # path='/data1/wyj/M/datasets/ccRCC_TCGA/data/'+file+'/'+xml
                # Tree=parse(path)
                root=Tree.documentElement

                #已经获得.xml文件树，.svs文件位置，下一步只需要：

                #裁切图片并存储到相应的文件夹
                slide = openslide.open_slide(slidepath)
                [w, h] = slide.level_dimensions[0]
                downscale = h / 2000
                annos=root.getElementsByTagName('Annotation')   #获取所有标记为'Annotataion'的文档部分
                TEMP_EXIST_COLORS=len(annos)            #获取annos的元素的数量，即有几种颜色的标记

                for ann in annos:   #本循环完成了把一个.xml文件中的所有标注线全部画出来
                    COLOR=ann.getAttribute('LineColor')

                    regs = ann.getElementsByTagName('Regions')
                    reg = regs[0].getElementsByTagName('Region')

                    for regid in range(len(reg)):
                        # print('Color:{}'.format(COLOR))
                        vs=reg[regid].getElementsByTagName('Vertices')
                        v=vs[0].getElementsByTagName('Vertex')
                        vx_array=[]
                        count = 0
                        for vertice in v:
                            try:
                                vx_array.append([float(vertice.getAttribute('X')),float(vertice.getAttribute('Y'))])
                                #count +=1
                                #print(count)
                            except:
                                pass
                        #print("PASS")
                        vx_array=np.array(vx_array)
                        CONTOUR = Path(vx_array)
                        bbox = np.array(CONTOUR.get_extents(), dtype=float)
                        x1, y1, x2, y2 = bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1]
                        vx_array_x = vx_array[:, 0] - x1
                        vx_array_y = vx_array[:, 1] - y1
                        try:
                            poly_contour = Polygon(vx_array - [x1, y1] + [bis, bis]).buffer(0.001)
                        except:
                            print('cant create polygon,only two points?')
                            continue

                        TEMP_polygom = Polygon(vx_array).buffer(0.001)  
                        ALL_POLYGON_OF_SWITCH[int(DICT[COLOR])].append(TEMP_polygom)
                        #SWITCH[COLOR][0]是一个二级索引，最终的结果是分类标签，我这里可以通过键值对直接改掉

                        # print('Color:{}'.format(COLOR))
                        plotx = vx_array[:, 0] / downscale
                        ploty = vx_array[:, 1] / downscale
                        plt.plot(plotx, ploty, int(DICT[COLOR]), linewidth=0.5)


                UNION_POLYGON_OF_SWITCH=[]
                for polyset in ALL_POLYGON_OF_SWITCH:    #ALL_POLYGON_OF_SWITCH：一个大的集合，包括0~6共7类，每类中含有多个多边形变量
                    UNION_POLY = union_of_polygons(polyset)      #UNION_POLY：每类多边形的集合
                    UNION_POLYGON_OF_SWITCH.append(UNION_POLY)          #UNION_POLYGON_OF_SWITCH：UNION_POLY的集合
                if TO_SHOW:
                    plt.savefig('TOSHOW3/{}_violet{}_g{}_b{}_r{}_y{}_k{}_w{}.png'.format(xml[:-8],UNION_POLYGON_OF_SWITCH[0].area,UNION_POLYGON_OF_SWITCH[1].area,UNION_POLYGON_OF_SWITCH[2].area,
                    UNION_POLYGON_OF_SWITCH[3].area,UNION_POLYGON_OF_SWITCH[4].area,UNION_POLYGON_OF_SWITCH[5].area, UNION_POLYGON_OF_SWITCH[6].area))
                
                xn = w // 224
                yn = h // 224
                #ACCEPTED_IM_NUMS=np.zeros((7),dtype=np.int)

                for x_ in range(int(xn)):
                    for y_ in range(int(yn)):
                        this_polygon = Polygon(
                            [(x_ * 224 + bis, y_ * 224 + bis), (x_ * 224 + 224 + bis, y_ * 224 + bis),
                             (x_ * 224 + 224 + bis, y_ * 224 + 224 + bis), (x_ * 224 + bis, y_ * 224 + 224 + bis)])
                        if SAVEABLE:
                            LABEL_of_P = label_of_this_polygon(this_polygon, UNION_POLYGON_OF_SWITCH)   #获得那个数字
                            #print(LABEL_of_P)
                            savp = train_or_test[0].format(LABEL_of_P) + xml[:-8] + '_x{}_y{}_{}.png'.format(  #感觉这里的-8有点问题，但不用改
                               int(x_ * 224), int(y_ * 224), im_num)
                            #print(savp)
                            tile = numpy.array(
                                slide.read_region(((x_ * 224), int(y_ * 224)), 0, (224, 224)))
                            
                            try:
                                io.imsave(savp, tile)
                                #print(savp)
                                #print(tile)
                                im_num = im_num + 1
                            except:
                                continue
                            #ACCEPTED_IM_NUMS[LABEL_of_P]+=1
                            #ACCEPTED_CLASS_VOLUME[LABEL_of_P]+=1
                            '''
                            
                            imageio.imwrite(savp, tile)  #如果直接这样保存，会保存很多低对比度的图片，也就是无效的区域
                            '''
                            #if TO_SHOW_PLT:
                            #    plt.gca().add_patch(plt.Rectangle(
                            #        xy=(x_ * (224) / downscale, y_ * (224) / downscale),
                            #        width=(224) / downscale,
                            #        height=(224) / downscale,
                            #        #edgecolor=ALL_CATS[LABEL_of_P],
                            #        fill=False, linewidth=1))
                #if TO_SHOW_PLT:
                #    plt.show()
                #plt.close()

