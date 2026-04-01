# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn import datasets
from sklearn.manifold import TSNE

cpx=0
cpy=0
MUTIP=1
MARKERSIZE=25*MUTIP
LINEWIDTH=1
width=4*4
height=3*4
dpi=300
fontsize=5*MUTIP
def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features

def get_distinct_colors(n,seed=4):
    np.random.seed(seed)
    cmap = plt.get_cmap('hsv')
    # 生成 n 个等间距的颜色
    colors = [cmap(i / (n - 1)) for i in range(n)]
    # np.random.shuffle(colors)
    # 将颜色转换为 RGB 格式
    rgb_colors = [(color[0], color[1], color[2]) for color in colors]
    return rgb_colors
CLASS_COLORS=get_distinct_colors(80+2)
CLASS_COLORS[14]=(0,1,0)
CLASS_COLORS[61]+=(0.2,)
CLASS_COLORS[59]=(0.2, 0.6, 0.2)
def plot_embedding(data, label, title,max_class_points=-1,centre_class=-1):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    # ax = plt.subplot(111)
    if max_class_points>0:
        class_counter=np.zeros((81,))
    for i in range(data.shape[0]):
        if max_class_points>0:
            class_counter[abs(int(label[i]))] += 1
            if class_counter[abs(int(label[i]))]>max_class_points:
                continue
        plt.text(data[i, 0], data[i, 1], str(int(label[i])),
                 color=CLASS_COLORS[abs(int(label[i]))],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    #plt.show()
    # plt.savefig(self.cfg.OUTPUT_DIR+'/tsne.jpg')
    return fig
BISE=-0.008
def plot_scatter(data, label, title,max_class_points=-1,centre_class=None,Board=0.15,box_subBoard=-0.15,O_A_B_point=[11,12,18,21,5,6,22,8],title_classname=True):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()
    O_A_B_point = [11, 12, 18, 21, 5, 6, 22, 8]
    #              *        o diam squ ^  v   6
    O_A_B_point = [11, 21, 18, 8, 37, 6, 26,33]#25
    # O_A_B_point = [11,34,35,36,37,38,39,40]
    if title_classname:
        titled_class=[]
    if centre_class is not None:
        cx,cy=data[label==centre_class,:][O_A_B_point[0]]
        ix, iy = data[label == centre_class, :][O_A_B_point[1]]
        ix2, iy2 = data[label == centre_class, :][O_A_B_point[2]]
        cpx=cx#(cx+ix2)/2
        cpy=cy-BISE#(cy+iy2)/2
        if box_subBoard < 0:
            plt.xlim(cpx-0.8*Board,cpx+1.2*Board)
            plt.ylim(cpy-1.15*Board,cpy+0.85*Board)
        else:
            subbx1,subbx2=(cpx - 0.7*Board, cpx + Board*0.8)
            subby1,subby2=(cpy - 1.1*Board, cpy + 0.4*Board)
            plt.xlim(subbx1,subbx2)
            plt.ylim(subby1,subby2)
        if box_subBoard<0:
            plt.plot(cx,cy,color=CLASS_COLORS[abs(int(centre_class))][:3],marker='*',markersize=MARKERSIZE)
            # plt.plot(ix,iy,color=CLASS_COLORS[abs(int(centre_class))], marker='o', markersize=MARKERSIZE*0.7)
            plt.plot(ix2, iy2, color=CLASS_COLORS[abs(int(centre_class))][:3], marker='o', markersize=MARKERSIZE*0.7)
            plt.plot((cx, ix2), (cy, iy2), color='r', linewidth=3, )
            # plt.plot((cx,ix), (cy,iy), color='r',linestyle='--',linewidth=LINEWIDTH)
            MARKERS=['d','s','^','v','h']
            for id,point in enumerate(O_A_B_point[3:]):
                if id ==4:
                    ixk, iyk = (0.45,0.67)
                else:
                    ixk,iyk= data[label == centre_class, :][point]
                plt.plot(ixk, iyk, color=CLASS_COLORS[abs(int(centre_class))][:3], marker=MARKERS[id], markersize=MARKERSIZE*0.7)
                plt.plot((cx, ixk), (cy, iyk), color='r',linewidth=3,linestyle='--',)
    else:
        centre_class=57
    COLOR_ARRAY=[]
    for i in range(len(label)):
        COLOR_ARRAY.append(CLASS_COLORS[abs(int(label[i]))])

    # ax = plt.subplot(111)
    if max_class_points>0:
        class_counter=np.zeros((81,))
    for i in range(data.shape[0]):
        if max_class_points>0:
            class_counter[abs(int(label[i]))] += 1
            if class_counter[abs(int(label[i]))]>max_class_points:
                continue
        if box_subBoard > 0:
            S_=  0.3
        else:
            S_ = 1
        plt.scatter(data[i, 0], data[i, 1], c=COLOR_ARRAY[i],s=S_)
        datax=data[i, 0]
        datay=data[i, 1]
        if title_classname :
            import random
            if random.random()<0.02:
                plt.text(data[i, 0], data[i, 1],'"{}"'.format(label[i]),c=COLOR_ARRAY[i],)
                titled_class.append(label[i])
        # if title_classname and label[i] not in titled_class \
        #         and datax>cpx-Board and datax<cpx+Board and datay>cpy-Board and datay<cpy+Board :
        #     plt.text(data[i, 0], data[i, 1],'"{}"'.format(label[i]),c=COLOR_ARRAY[i])
        #     titled_class.append(label[i])
    if box_subBoard>0:
        plt.gca().add_patch(
            plt.Rectangle(xy=(cpx-0.8*box_subBoard,cpy-1.15*box_subBoard), width=2*box_subBoard, height=2*box_subBoard, edgecolor='k',
                      fill=False, linewidth=1,linestyle='--' ))
    plt.xticks([])
    plt.yticks([])
    # plt.text(data[1, 0], data[1, 1],'bottle', c=COLOR_ARRAY[1])
    # plt.text(data[2, 0], data[2, 1],'potted plant', c=COLOR_ARRAY[2])
    if box_subBoard <= 0:
        # plt.text(0.34, 0.1, 'bottle', c=COLOR_ARRAY[1])
        # plt.text(0.45, 0.4, 'cup', c=CLASS_COLORS[2])
        # plt.text(0.56, 0.388, 'dining table', c=CLASS_COLORS[58])
        # plt.text(0.42, 0.18, 'potted plant', c=CLASS_COLORS[57])
        # MARKERS = ['*','o','d', 's', '^', 'v', 'h']
        # MARK_NAME=['0.Original Text',
        #     '1.Original Visual',
        #            '2.Fully-Finetune',
        #            '3.Linear Probing',
        #            '4.Promt Tuning-MaPLe',
        #            '5.MQ-Det',
        #            '6.VisTex-GLIP',
        #            ]

        #
        # from pylab import mpl
        # # 设置显示中文字体
        # mpl.rcParams["font.sans-serif"] = ["SimHei"]
        # mpl.rcParams["axes.unicode_minus"] = False
        # plt.text(0.43, 0.92, 'airplane', c=CLASS_COLORS[1])
        plt.text(0.62, 0.87, 'dining table', c=CLASS_COLORS[59])
        plt.text(0.35, 0.73, 'cat', c=CLASS_COLORS[14])
        plt.text(0.5, 0.85, 'car', c=CLASS_COLORS[61][:3])
        MARKERS = ['*','o','d', 's', '^', 'v', 'h']
        MARK_NAME=['0.text',
            '1.GLIP 0-shot',
                   '2.VPT',
                   '3.CoOp',
                   '4.LoRA',
                   '5.MaPLe',
                   '6.Ours',
                   ]
        FIG_PIN_x=0.33
        FIG_PIN_y=0.879
        wide=0.01
        height=0.019
        PIN_MARKERSIZE=6*MUTIP
        for id,marker in enumerate(MARKERS):
            plt.gca().add_patch(
                plt.Rectangle(xy=(FIG_PIN_x-0.048,FIG_PIN_y-0.125), width=0.12,#0.140,
                              height=0.180, edgecolor='k',
                              fill=True, linewidth=1,facecolor='white'))
            plt.plot(FIG_PIN_x+0.002, FIG_PIN_y-height*id, color=CLASS_COLORS[abs(int(centre_class))][:3], marker=marker, markersize=PIN_MARKERSIZE)
            plt.text(FIG_PIN_x+wide, FIG_PIN_y-height*id-0.005, MARK_NAME[id], c='k',fontsize=9)
    # plt.text(data[1, 0], data[1, 1],'bottle', c=COLOR_ARRAY[1])
    # plt.text('"potted plant"', x=data[2, 0], y=data[2, 1], c=COLOR_ARRAY[2])
    # plt.text('"cup"', x=data[3, 0], y=data[3, 1], c=COLOR_ARRAY[3])
    # plt.title(title)
    # #plt.show()
    # plt.savefig(self.cfg.OUTPUT_DIR+'/tsne.jpg')
    return
def plot_scatter_rightdown(data, label, title,max_class_points=-1,centre_class=None,Board=0.1,box_subBoard=-0.1):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()
    if centre_class is not None:
        cx,cy=data[label==centre_class,:][5]
        ix, iy = data[label == centre_class, :][9]
        plt.xlim(cx-Board,cx+Board)
        plt.ylim(cy-Board,cy+Board)
        plt.plot(cx,cy,color=CLASS_COLORS[abs(int(centre_class))],marker='*',markersize=12)
        plt.plot(ix, iy, color=CLASS_COLORS[abs(int(centre_class))], marker='+', markersize=12)
        plt.plot((cx,ix), (cy,iy), color='r')
    COLOR_ARRAY=[]
    for i in range(len(label)):
        COLOR_ARRAY.append(CLASS_COLORS[abs(int(label[i]))])

    # ax = plt.subplot(111)
    if max_class_points>0:
        class_counter=np.zeros((81,))
    for i in range(data.shape[0]):
        if max_class_points>0:
            class_counter[abs(int(label[i]))] += 1
            if class_counter[abs(int(label[i]))]>max_class_points:
                continue
        plt.scatter(data[:, 0], data[:, 1], c=COLOR_ARRAY,s=1)
    if box_subBoard>0:
        plt.gca().add_patch(
            plt.Rectangle(xy=(cx-box_subBoard, cy-box_subBoard), width=2*box_subBoard, height=2*box_subBoard, edgecolor='k',
                      fill=False, linewidth=1,linestyle='--' ))
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(title)
    # #plt.show()
    # plt.savefig(self.cfg.OUTPUT_DIR+'/tsne.jpg')
    return
# def plot_embedding(data, label, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     for i in range(data.shape[0]):
#         plt.text(data[i, 0], data[i, 1], str(label[i]),
#                  color=plt.cm.Set1(label[i] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title)
#     return fig


def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    # plt.imshow(fig)
    plt.savefig('tsne.jpg')

def classid_of_word(word,split_class):
    for id,a_class in enumerate(split_class):
        if word in a_class:
            return id
def select_names_of_array(names,data_array,split_class):
    keep_mask=data_array<-100
    for name in names:
        classid=-(classid_of_word(name,split_class)+1)
        keep_mask+=(data_array==classid)
    return keep_mask

from collections import Counter
import heapq


def top_n_elements(label_array, n=20):
    # 统计每个元素的出现次数
    counter = Counter(label_array)

    # 使用 heapq.nlargest 找到出现次数最多的前 n 个元素
    # heapq.nlargest 的第二个参数是一个元组 (count, label)，按 count 降序排列
    most_common_elements = heapq.nlargest(n, counter.items(), key=lambda item: item[1])

    # 提取元素标签并组成新的数组
    top_elements = [label for label, count in most_common_elements]

    return top_elements
def select_top_n_of_array(data_array,n=20):
    keep_mask=data_array<-1000
    top_n_elem=top_n_elements(data_array,n)
    for elem in top_n_elem:
        keep_mask+=(data_array==elem)
    return keep_mask

# 示例用法
# label_array = [1, 2, 3, 1, 2, 2, 4, 5, 1, 6, 7, 8, 9, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9,9,9, 0]
# top_elements = top_n_elements(label_array,n=2)
# print(top_elements)
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    all_prompt='person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush'
    split_class = all_prompt.split('.')
    selected_class_set = [
        'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
        'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']
    for id, item in enumerate(split_class):
        split_class[id] = item.strip()
    # selected_class_set = list(set(split_class) - set(selected_class_set))
    # selected_one_class = ['cup']  # ['potted plant']
    # map = io.imread('Fullfinetune.jpg')
    # io.imsave('Fullfinetune2.jpg', map[170:1288, 236:1733, :])

    # all_array=np.load("/home/data/jy/GLIP/OUTPUT_TSNE_our0/ours_feature.npz")
    # data=all_array['arr_0']
    # label=all_array['arr_1']
    # # fig, ax1 = plt.subplots()
    # # result2 = tsne.fit_transform(data[label < 0, :])
    # # plot_scatter(result2,
    # #              label[label<0],
    # #              't-SNE embedding of the digits (time %.2fs)'
    # #              % (time() - t0), max_class_points=-1)
    # # plt.show()
    # # pass
    # fig, ax1 = plt.subplots()
    # OTHER_POINTS=[9,10,11]
    # resultx = tsne.fit_transform(data[label < -40, :])
    # plot_scatter(resultx,
    #                    label[label < -40],
    #                    't-SNE embedding of the digits (time %.2fs)'
    #                    % (time() - t0), max_class_points=-1,centre_class=-57,Board=0.1,O_A_B_point=[9,11,10])
    # insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    # plt.xticks([])
    # plt.yticks([])
    # plot_scatter(resultx,
    #                    label[label < -40],
    #                    't-SNE embedding of the digits (time %.2fs)'
    #                    % (time() - t0), max_class_points=-1, centre_class=-57,Board=0.5,box_subBoard=0.1,O_A_B_point=[9,11,10])
    # # plt.title('ourA')
    # plt.savefig('LINEAR_PROBE.jpg')
    # plt.show()
    #
    #
    # fig, ax1 = plt.subplots()
    # resultx = tsne.fit_transform(data[select_names_of_array(selected_class_set, label, split_class), :])
    # plot_scatter(resultx,
    #              label[select_names_of_array(selected_class_set, label, split_class)],
    #              't-SNE embedding of the digits (time %.2fs)'
    #              % (time() - t0), max_class_points=-1, centre_class=-57, Board=0.12,O_A_B_point=OTHER_POINTS)
    # insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    # plt.xticks([])
    # plt.yticks([])
    # plot_scatter(resultx,
    #              label[select_names_of_array(selected_class_set, label, split_class)],
    #              't-SNE embedding of the digits (time %.2fs)'
    #              % (time() - t0), max_class_points=-1, centre_class=-57, Board=0.5, box_subBoard=0.12,O_A_B_point=OTHER_POINTS)
    # # plt.title('ourA')
    # plt.savefig('OUR.jpg')
    # #plt.show()
    # all_array = np.load('../consep_feature.npz')
    # data=all_array['arr_0']
    # label=all_array['arr_1']
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=7, random_state=42)
    # kmeans.fit(data)
    # labels = kmeans.labels_
    # label60=labels[:60]
    # centers = kmeans.cluster_centers_
    # print("Cluster centers:")
    # print(centers)
####################################################
    all_array = np.load("/home/data/jy/GLIP/OUTPUT_TSNE_0/ours_feature.npz")
    data=all_array['arr_0']
    label=all_array['arr_1']

    # result = tsne.fit_transform(data)
    # fig = plot_embedding(result[label<0,:], label[label<0],
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0),max_class_points=1000)
    # fig = plot_embedding(result[(label == -27) + (label == -40) + (label == -59) + (label == -68) + (label == -74), :],
    #                      label[(label == -27) + (label == -40) + (label == -59) + (label == -68) + (label == -74)],
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0), max_class_points=-1)
    # fig = plot_embedding(result[select_names_of_array(selected_class_set, label, split_class), :],
    #                      label[select_names_of_array(selected_class_set, label, split_class)],
    #                      't-SNE embedding of the digits (time %.2fs)'
    #                      % (time() - t0), max_class_points=-1)

    # fig, ax1 = plt.subplots()
    # resultx = tsne.fit_transform(data[label < -55, :])
    # plot_scatter(resultx,
    #                    label[label < -55],
    #                    't-SNE embedding of the digits (time %.2fs)'
    #                    % (time() - t0), max_class_points=-1,centre_class=-57,Board=0.15)
    # insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    # plt.xticks([])
    # plt.yticks([])
    # plot_scatter(resultx,
    #                    label[label < -55],
    #                    't-SNE embedding of the digits (time %.2fs)'
    #                    % (time() - t0), max_class_points=-1, centre_class=-57,Board=0.5,box_subBoard=0.15)
    # # plt.title('ourA')
    # plt.savefig('only_train_adapter.jpg')
    # plt.show()
    plot_TSNE=False
    if plot_TSNE:
        fig = plt.gcf()
        INCH = 10
        fig.set_size_inches(0.64* INCH, 0.48 * INCH)
        fig, ax1 = plt.subplots(figsize=(0.64* INCH, 0.48 * INCH),dpi=dpi)
        # resultx = tsne.fit_transform(data[label<0])#[select_names_of_array(selected_class_set, label, split_class), :])
        labelx=label[label<0]#[select_names_of_array(selected_class_set, label, split_class)]
        uniq_label,label_count=np.unique(label,return_counts=True)
        uniq_count=np.stack((uniq_label,label_count))
        CENTRE_CLASS=[-61,-59,-27,-1]
        indices = np.argsort(label)
        datax=data[label<0]
        resultx1=datax[select_top_n_of_array(labelx)]
        labelx1 = labelx[select_top_n_of_array(labelx)]
        resultx = tsne.fit_transform(resultx1)
        plot_scatter(resultx,
                           labelx1,
                           't-SNE embedding of the digits (time %.2fs)'
                           % (time() - t0), max_class_points=-1,centre_class=-61,Board=0.18,title_classname=False)
        insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
        # plt.xticks([])
        # plt.yticks([])
        plot_scatter(resultx,
                           labelx1,
                           't-SNE embedding of the digits (time %.2fs)'
                           % (time() - t0), max_class_points=-1, centre_class=-61,Board=0.7,box_subBoard=0.18,title_classname=False)
        # plt.title('ourA')
        plt.savefig('TPAMI_TSNE.jpg')
        plt.show()
        # map = io.imread('TPAMI_TSNE.jpg')
        # io.imsave('TPAMI_TSNE.jpg', map[170:1288, 236:1733, :])
        # label_SE=label[select_names_of_array(selected_class_set, label, split_class)]
        # result57=resultx[label_SE==-57]
        # result40 = resultx[label_SE == -40]
        # all_array = np.load("/home/data/jy/GLIP/OUTPUT_TSNE_ours/ours_feature.npz")
        # all_array = np.load("/home/data/jy/GLIP/OUTPUT_TSNE_others/ours_feature.npz")
        # data=all_array['arr_0']
        # label=all_array['arr_1']
    # MARK_NAME=[
    #     '1.Ground Truth',
    #            '2.GLIP 0-shot',
    #            '3.CoOp',
    #            '4.LoRA',
    #            '5.MaPLe',
    #            '6.Ours',
    #            ]
    MARK_NAME=[#'0.Original Text',
        '1.Original GLIP',
               '2.Fully-Finetune',
               '3.Linear Probing',
               '4.Promt Tuning-MaPLe',
               '5.MQ-Det',
               '6.VisTex-GLIP',
               ]
    COLOR_SET=['blue','purple','pink','cyan','green','red']
    from collections import Counter
    from scipy.stats import gaussian_kde
    plt.figure(figsize=(10, 6))
    fig,ax=plt.subplots()
    # 绘制每一组的频度曲线
    targets=[-61,-59,-57,-56,-47,-27,]
    targets = [8,2,3,11,6,7 ]
    targets = [8,13,12,15,9,17]
    # targets = [7,2, ]
    # targets=[-51,-50,-49,-48,-47,-46]#-61, -59, -57, -27, -1]-56,-47
    for i ,tar in enumerate(targets):
        vectors=data[label==tar]
        center = vectors[0,:]
        # 计算余弦相似度
        cosine_similarities = np.dot(vectors, center) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(center))
        #cosine_similarities = np.dot(vectors, center) / (np.linalg.norm(center) * np.linalg.norm(center))
        distances = np.linalg.norm(vectors - center, axis=1)
        # 由于余弦相似度已经在-1到1之间，且通常不会是负的，我们不需要归一化到0-1
        # 但为了符合你的要求，我们可以简单地将相似度平移和缩放到0-1（如果它们都是非负的）
        # cosine_similarities = (cosine_similarities - cosine_similarities.min()) / (cosine_similarities.max() - cosine_similarities.min())
        # bins = np.linspace(0, 1, 100)  # 100个bin来平滑曲线
        # hist, bin_edges = np.histogram(cosine_similarities, bins=bins)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # 注意：上面的归一化步骤在余弦相似度都是非负的情况下才是有效的。如果有可能出现负值，
        # 你可能需要重新考虑是否要对相似度进行这样的处理，或者如何处理负值。

        # 在这个例子中，我们假设所有相似度都是非负的，并且不进行上面的归一化步骤。

        # 生成频度分布（使用直方图）并进行平滑（使用KDE）
        kde = gaussian_kde(cosine_similarities, bw_method=0.4)  # 可以调整bw_method来控制平滑程度
        x_values = np.linspace(0.4, 1, 100)  # KDE的x轴值，这里我们假设相似度都在0-1之间
        y_values = kde(x_values) # KDE的y轴值（概率密度）

        # 为了得到占总特征矢量数的百分比，我们需要对y_values进行积分并归一化。
        # 但由于我们使用的是KDE，y_values已经是概率密度，所以我们可以直接绘制它，
        # 并理解其下的面积（如果积分的话）应该接近1（但不是严格的1，因为KDE是估计）。
        # 如果要得到严格的百分比，我们需要对直方图进行归一化，而不是对KDE。

        # 绘制图形
        ax.plot(x_values, y_values, label=MARK_NAME[i],linewidth=1,c=COLOR_SET[i])
        ax.fill_between(x_values, y_values, where=x_values >= 0, alpha=0.5,color=COLOR_SET[i])
        uneven_ticks = [0.4 , 0.6,  0.8,  1. ,]  # 这里我们手动选择了刻度位置
        ax.set_xticks(uneven_ticks)
        ax.set_ylim(0, 8)
        # 调整x轴刻度，使0-0.4区间更短
        # ax.set_xticks(np.linspace(0, 1, 11), np.round(np.linspace(0, 1, 11), 2))  # 默认刻度
        # ax.xaxis.set_tick_params(which='major',
        #                          length=[0.05 if 0 <= x <= 0.4 else 0.1 for x in ax.get_xticks()])
        # ax.set_tick_params(which='major',
        #                           length=[0.05 if 0 <= x <= 0.4 else 0.1 for x in plt.gca().get_xticks()])
        # 注意：上面的刻度调整方法可能不是最优的，因为它依赖于当前的刻度位置和数量。
        # 你可能需要根据你的具体需求调整这部分代码。

        # 设置标签和标题
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Percentage')
        plt.title('Cosine Similarity Frequency Distribution')
        plt.legend()
        plt.grid(True)

        # 显示图形（为了示例清晰，这里注释掉，但在实际代码中应该取消注释）
        # plt.show()
    plt.show()  # 显示图形





    targets=[-61,-59,-57,-27,-1]
    for tid ,tar in enumerate(targets):
        vectors=data[label==tar]
        center_vector=vectors[0,:]
        cosine_similarities = np.dot(vectors, center_vector) / (
                np.linalg.norm(vectors, axis=1) * np.linalg.norm(center_vector)
        )
        bins = np.linspace(0, 1, 100)  # 100个bin来平滑曲线
        hist, bin_edges = np.histogram(cosine_similarities, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 使用高斯核密度估计来平滑频度曲线
        kde = gaussian_kde(cosine_similarities, bw_method=0.1)  # 可以调整bw_method来控制平滑程度
        x_values = np.linspace(0, 1, 10000)  # 用于绘制KDE曲线的x值
        kde_value=kde(x_values)
        y_values = kde(x_values)   #y_values = kde(x_values)

        # 绘制频度直方图和KDE曲线
        plt.bar(bin_centers, hist / len(vectors) * bin_edges[1] - bin_edges[0], width=bin_edges[1] - bin_edges[0],
                alpha=0.5, label='Histogram', edgecolor='black')
        plt.plot(x_values, y_values, label='label{}'.format(tar), color='red')
    plt.xlabel('Cosine Similarity (scaled to 0-10)')
    plt.ylabel('Frequency')
    plt.title('Cosine Similarity Frequency Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

    fig, ax1 = plt.subplots()
    resultx = tsne.fit_transform(data[label < 0, :])

    plot_scatter(resultx,
                       label[label < 0],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1,centre_class=-61,Board=0.15)
    insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    plt.xticks([])
    plt.yticks([])
    plot_scatter(resultx,
                       label[label < 0],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1, centre_class=-61,Board=0.5,box_subBoard=0.15)
    # plt.title('ourA')
    plt.savefig('MQDET.jpg')
    #plt.show()


    fig, ax1 = plt.subplots()
    result2 = tsne.fit_transform(data[label < 0, :])
    plot_scatter(result2,
                 label[label<0],
                 't-SNE embedding of the digits (time %.2fs)'
                 % (time() - t0), max_class_points=-1, centre_class=-57,Board=0.1)
    # insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    # plt.xticks([])
    # plt.yticks([])
    plot_scatter(result2,
                 label[label<0],
                 't-SNE embedding of the digits (time %.2fs)'
                 % (time() - t0), max_class_points=-1, Board=0.5)
    #plt.show()

    fig, ax1 = plt.subplots()
    resultx = tsne.fit_transform(data[select_names_of_array(selected_class_set, label, split_class), :])
    plot_scatter(resultx,
                       label[select_names_of_array(selected_class_set, label, split_class)],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1,centre_class=-57)
    insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    plt.xticks([])
    plt.yticks([])
    plot_scatter(resultx,
                       label[select_names_of_array(selected_class_set, label, split_class)],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1, centre_class=-57,Board=0.5)
    #plt.show()

    fig, ax1 = plt.subplots()
    resultx = tsne.fit_transform(data[select_names_of_array(selected_class_set, label, split_class), :])
    plot_scatter(resultx,
                       label[select_names_of_array(selected_class_set, label, split_class)],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1,centre_class=-57)
    insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    plt.xticks([])
    plt.yticks([])
    plot_scatter(resultx,
                       label[select_names_of_array(selected_class_set, label, split_class)],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1, centre_class=-57,Board=0.5)
    #plt.show()

    f = plot_scatter(resultx,
                       label[select_names_of_array(selected_class_set, label, split_class)],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1,centre_class=-57)

    result2 = tsne.fit_transform(data[label<0,:])
    f = plot_embedding(result2, label[label<0],
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0),max_class_points=1000)
    result3 = tsne.fit_transform(
        data[(label == -27) + (label == -40) + (label == -59) + (label == -68) + (label == -74), :])
    fig3 = plot_embedding(result3,
                         label[(label == -27) + (label == -40) + (label == -59) + (label == -68) + (label == -74)],
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0), max_class_points=1000)

    result4 = tsne.fit_transform(
        data[(label == -27) + (label == -40) + (label == -59) + (label == -68) + (label <= -74), :])
    fig4 = plot_embedding(result4,
                         label[(label == -27) + (label == -40) + (label == -59) + (label == -68) + (label <= -74)],
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0), max_class_points=1000)
    fig, ax1 = plt.subplots()
    insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    pass
