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

def get_distinct_colors(n,seed=42):
    np.random.seed(seed)
    cmap = plt.get_cmap('hsv')
    # 生成 n 个等间距的颜色
    colors = [cmap(i / (n - 1)) for i in range(n)]
    # np.random.shuffle(colors)
    # 将颜色转换为 RGB 格式
    rgb_colors = [(color[0], color[1], color[2]) for color in colors]
    return rgb_colors
CLASS_COLORS=get_distinct_colors(80+1)
CLASS_COLORS[58]=(0,1,0)
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
def plot_scatter(data, label, title,max_class_points=-1,centre_class=None,Board=0.15,box_subBoard=-0.15,O_A_B_point=[5,9,10,11,13,14,15,16],title_classname=True):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    # fig = plt.figure()
    if title_classname:
        titled_class=[]
    if centre_class is not None:
        cx,cy=data[label==centre_class,:][O_A_B_point[0]]
        ix, iy = data[label == centre_class, :][O_A_B_point[1]]
        ix2,iy2=(0.53,0.26)# ix2, iy2 = data[label == centre_class, :][O_A_B_point[2]]
        cpx=cx#(cx+ix2)/2
        cpy=cy-BISE#(cy+iy2)/2
        if box_subBoard < 0:
            plt.xlim(cpx-Board,cpx+Board)
            plt.ylim(cpy-Board,cpy+Board)
        else:
            plt.xlim(cpx - Board, cpx + Board*1.2)
            plt.ylim(cpy - 0.8*Board, cpy + 1.4*Board)
        if box_subBoard<0:
            plt.plot(cx,cy,color=CLASS_COLORS[abs(int(centre_class))],marker='*',markersize=MARKERSIZE)
            # plt.plot(ix,iy,color=CLASS_COLORS[abs(int(centre_class))], marker='o', markersize=MARKERSIZE*0.7)
            plt.plot(ix2, iy2, color=CLASS_COLORS[abs(int(centre_class))], marker='o', markersize=MARKERSIZE*0.7)
            plt.plot((cx, ix2), (cy, iy2), color='r', linewidth=3, )
            # plt.plot((cx,ix), (cy,iy), color='r',linestyle='--',linewidth=LINEWIDTH)
            MARKERS=['d','s','^','v','h']
            for id,point in enumerate(O_A_B_point[3:]):
                ixk,iyk= data[label == centre_class, :][point]
                plt.plot(ixk, iyk, color=CLASS_COLORS[abs(int(centre_class))], marker=MARKERS[id], markersize=MARKERSIZE*0.7)
                plt.plot((cx, ixk), (cy, iyk), color='r',linewidth=3,linestyle='--',)
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
        # if title_classname and label[i] not in titled_class \
        #         and datax>cpx-Board and datax<cpx+Board and datay>cpy-Board and datay<cpy+Board :
        #     plt.text(data[i, 0], data[i, 1],'"{}"'.format(label[i]),c=COLOR_ARRAY[i])
        #     titled_class.append(label[i])
    if box_subBoard>0:
        plt.gca().add_patch(
            plt.Rectangle(xy=(cpx-box_subBoard, cpy-box_subBoard), width=2*box_subBoard, height=2*box_subBoard, edgecolor='k',
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
        plt.text(0.34, 0.1, '瓶子', c=COLOR_ARRAY[1])
        plt.text(0.45, 0.4, '杯子', c=CLASS_COLORS[2])
        plt.text(0.56, 0.388, '餐桌', c=CLASS_COLORS[58])
        plt.text(0.42, 0.18, '盆栽', c=CLASS_COLORS[57])
        MARKERS = ['*','o','d', 's', '^', 'v', 'h']
        MARK_NAME=['0.原始文本',
            '1.原始图像',
                   '2.全微调',
                   '3.线性探测',
                   '4.提示微调',
                   '5.多模态提问',
                   '6.视觉文本化',
                   ]
        FIG_PIN_x=0.262
        FIG_PIN_y=0.43-BISE
        wide=0.01
        height=0.019
        PIN_MARKERSIZE=6*MUTIP
        for id,marker in enumerate(MARKERS):
            plt.gca().add_patch(
                plt.Rectangle(xy=(0.25,0.308), width=0.12,#0.140,
                              height=0.180, edgecolor='k',
                              fill=True, linewidth=1,facecolor='white'))
            plt.plot(FIG_PIN_x+0.002, FIG_PIN_y-height*id, color=CLASS_COLORS[abs(int(centre_class))], marker=marker, markersize=PIN_MARKERSIZE)
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
    selected_one_class=['potted plant']
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
    all_array = np.load('/home/data/jy/GLIP/consep_feature_base.npz')
    data=all_array['arr_0']
    label=all_array['arr_1']
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    label60=labels[:60]
    centers = kmeans.cluster_centers_
    print("Cluster centers:")
    print(centers)
    labelGT_KMEAN = np.concatenate(([label,], [labels,]), axis=0)
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

    fig = plt.gcf()
    INCH = 10
    fig.set_size_inches(0.64* INCH, 0.48 * INCH)
    fig, ax1 = plt.subplots(figsize=(0.64* INCH, 0.48 * INCH),dpi=dpi)
    resultx = tsne.fit_transform(data[select_names_of_array(selected_class_set, label, split_class), :])
    plot_scatter(resultx,
                       label[select_names_of_array(selected_class_set, label, split_class)],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1,centre_class=-57,Board=0.2)
    insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    # plt.xticks([])
    # plt.yticks([])
    plot_scatter(resultx,
                       label[select_names_of_array(selected_class_set, label, split_class)],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1, centre_class=-57,Board=0.5,box_subBoard=0.2)
    # plt.title('ourA')
    plt.savefig('Fullfinetune.jpg')
    plt.show()
    map = io.imread('Fullfinetune.jpg')
    io.imsave('Fullfinetune2.jpg', map[170:1288, 236:1733, :])
    label_SE=label[select_names_of_array(selected_class_set, label, split_class)]
    result57=resultx[label_SE==-57]
    result40 = resultx[label_SE == -40]
    # all_array = np.load("/home/data/jy/GLIP/OUTPUT_TSNE_ours/ours_feature.npz")
    # all_array = np.load("/home/data/jy/GLIP/OUTPUT_TSNE_others/ours_feature.npz")
    # data=all_array['arr_0']
    # label=all_array['arr_1']
    fig, ax1 = plt.subplots()
    resultx = tsne.fit_transform(data[label < 0, :])

    plot_scatter(resultx,
                       label[label < 0],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1,centre_class=-57,Board=0.15)
    insert = fig.add_axes([0.65, 0.11, 0.25, 0.25])
    plt.xticks([])
    plt.yticks([])
    plot_scatter(resultx,
                       label[label < 0],
                       't-SNE embedding of the digits (time %.2fs)'
                       % (time() - t0), max_class_points=-1, centre_class=-57,Board=0.5,box_subBoard=0.15)
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
