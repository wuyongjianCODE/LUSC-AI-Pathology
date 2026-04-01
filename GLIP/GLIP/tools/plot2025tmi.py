import matplotlib.pyplot as plt
import os
from skimage import io,transform
import cv2
import numpy as np
from pycocotools import mask as maskUtils
def print_self_training_COLORED_ALLSHOW():
    fid=0
    submits = ["/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/GT/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.28124/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.27039/",
                              "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.26471/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.24936/",#5
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.20576/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.26655/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.17041/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.16846/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15769/",#10
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15415/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10834/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08137/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10627/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08620/",#15
                              "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07695/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07728/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07802/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07383/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.06003/",
               '/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.01511/',
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.12396/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.13447/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15052/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.13447/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.14786/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.13349/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.12074/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.11317/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.16650/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10099/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08498/",
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        allpic = ["000000000186.jpg","000000000027.jpg",
                  "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
                  "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
                  "000000000456.jpg","000000000023.jpg",]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        # if image_id not in allpic:
        #     continue
        # NEEDTOSHOW=False
        # for picid in allpic:
        #     if 'IMG_{}'.format(picid) in image_id:
        #         NEEDTOSHOW=True
        #         SHORTID=picid
        #         rank=ranks[SHORTID]
        # if not NEEDTOSHOW:
        #     continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 2*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz123456789012345678901234567890'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[iid] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
                plt.title('({})'.format(iid), y=-0.1, color='k', fontsize=20)
            except:
                pass
            if fid == 3:
                plt.title('({})'.format(TITLE[iid]),y=-0.1,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW2025tmi/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW2025tmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_COLORED_ALLSHOWF():
    fid=0
    submits = ["/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/GT/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.28124/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.27039/",
                              "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.26471/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.24936/",#5
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.20576/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.26655/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.17041/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.16846/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15769/",#10
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15415/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10834/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08137/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10627/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08620/",#15
                              "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07695/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07728/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07802/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07383/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.06003/",
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000399.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        allpic = ["000000000186.jpg","000000000027.jpg",
                  "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
                  "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
                  "000000000456.jpg","000000000023.jpg",]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        # if image_id not in allpic:
        #     continue
        # NEEDTOSHOW=False
        # for picid in allpic:
        #     if 'IMG_{}'.format(picid) in image_id:
        #         NEEDTOSHOW=True
        #         SHORTID=picid
        #         rank=ranks[SHORTID]
        # if not NEEDTOSHOW:
        #     continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 2*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[iid] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                pass
            if fid == 3:
                plt.title('({})'.format(TITLE[iid]),y=-0.1,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW2025tmi/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW2025tmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def change_color(add_up,color0,colorout):
    out=add_up.copy()
    PLACE0=add_up[:,:,0]==color0[0]
    PLACE1 = add_up[:, :, 1] == color0[1]
    PLACE2 = add_up[:, :, 2] == color0[2]
    PLACE =PLACE0 *PLACE1*PLACE2
    out[:,:,0][PLACE]=colorout[0]
    out[:, :, 1][PLACE] = colorout[1]
    out[:, :, 2][PLACE] = colorout[2]
    return out
def print_self_training_COLORED_ALLSHOW2():
    fid=0
    submits = ["/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/GT/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.29917/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.28100/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26684/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26348/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26352/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26088/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25962/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25874/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25234/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25026/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24648/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24670/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24208/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24703/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.22126/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21579/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21159/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21069/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20528/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20370/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20845/",  # 5
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20853/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20049/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20012/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17865/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18738/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18573/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19977/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19304/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19288/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19793/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19656/",  # 10
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18700/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16703/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16396/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17952/"
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17682/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17158/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16196/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16359/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15400/",  # 15
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15822/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15543/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14110/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14720/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14961/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13597/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13568/",  # 20
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13161/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.12213/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11622/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.12044/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11042/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11751/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.09088/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.09521/",

               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        allpic = ["000000000186.jpg","000000000027.jpg",
                  "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
                  "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
                  "000000000456.jpg","000000000023.jpg",]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        # if image_id not in allpic:
        #     continue
        # NEEDTOSHOW=False
        # for picid in allpic:
        #     if 'IMG_{}'.format(picid) in image_id:
        #         NEEDTOSHOW=True
        #         SHORTID=picid
        #         rank=ranks[SHORTID]
        # if not NEEDTOSHOW:
        #     continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 20
        if fid % H == 0:
            plt.close()
        COL_NUMS = len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 2*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz12345678901234567890'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[iid] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
                plt.title('({})'.format(iid), y=-0.1, color='k', fontsize=20)
            except:
                pass
            # plt.title('({})'.format(TITLE[iid]),y=-0.1,color='k',fontsize=40)
            # print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW2025tmi2/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW2025tmi2/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_COLORED_ALLSHOW2final():
    fid=0
    submits = ["/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/GT/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.29917/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.28100/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26684/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26348/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26352/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26088/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25962/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25874/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25234/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25026/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24648/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24670/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24208/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24703/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.22126/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21579/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21159/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21069/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20528/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20370/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20845/",#5
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20853/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20049/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20012/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17865/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18738/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18573/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19977/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19304/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19288/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19793/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19656/",#10
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18700/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16703/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16396/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17952/"
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17682/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17158/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16196/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16359/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15400/",#15
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15822/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15543/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14110/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14720/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14961/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13597/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13568/",#20
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13161/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.12213/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11622/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.12044/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11042/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11751/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.09088/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.09521/",

               ]
    for image_id in os.listdir(submits[1]):
        allpic = [  "000000000717.jpg",]
        # allpic = ["000000000186.jpg","000000000027.jpg",
        #           "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
        #           "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
        #           "000000000456.jpg","000000000023.jpg",]
        # allpic = [3183, 2535, 8496]
        ranks={"000000000717.jpg":[
            55,46,30,29,
            54,52,32,20,
            34,31,22,12,
            42,33,28,27,
            44,41,18,17,
            48,45,39,35,
            24,28,15,16,
            25,20,11,4,
            14,10,5,3,
            6,13,7,1,
            0,0,0,0],}
        ranks={"000000000717.jpg":[
            55,54,34,42,44,41,5 ,25,14,6, 0,
            46,52,31,33,48,45,28,20,15,13,0,
            30,32,22,28,18,39,10,11, 24,7, 0,
            29,20,12,27,17,35, 3, 4,16,1, 0,
            ],}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        if image_id not in allpic:
            continue
        NEEDTOSHOW=False
        for picid in allpic:
            if '{}'.format(picid) in image_id:
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        # if not NEEDTOSHOW:
        #     continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 11
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 0.9*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz12345678901234567890'
        for iid in range(4*COL_NUMS):
            idx += 1
            plt.subplot(H, COL_NUMS, idx)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))

                # plt.title('({})'.format(iid), y=-0.1, color='k', fontsize=20)
                if iid==(4*COL_NUMS-2):
                    # plt.close()
                    im_method_GT = io.imread(submits[rank[0]] + '/' + image_id)[:, :, :]
                    im_method[210:255,50:150,:]=im_method_GT[210:255,50:150,:]
                    im_method_GT2 = io.imread(submits[rank[11]] + '/' + image_id)[:, :, :]
                    im_method[100:160,135:163,:]=im_method_GT2[100:160,135:163,:]
                    # plt.imshow(im_method);plt.show()
                    im_method[140:200, 161:220, :] = im_method_GT[140:200, 161:220, :]
                    # plt.imshow(im_method);plt.show()

                plt.imshow(im_method[:,:,:])
            except:
                pass
            if iid>=3*COL_NUMS:
                plt.title('({})'.format(TITLE[iid-3*COL_NUMS]),y=-0.168,color='k',fontsize=40)
            # print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW2025tmi2/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW2025tmi2/ALLcom.png')
    map=io.imread('TOSHOW2025tmi2/ALLcom.png')
    io.imsave('TOSHOW2025tmi2/ALLcom.png',map[:-300,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_COLORED_ALLSHOW2WITH_OR_WITHOUT():
    fid=0
    submits = ["/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/GT/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.29917/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.28100/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26684/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26348/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26352/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.26088/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25962/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25874/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25234/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.25026/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24648/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24670/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24208/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.24703/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.22126/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21579/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21159/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.21069/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20528/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20370/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20845/",#5
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20853/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20049/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.20012/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17865/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18738/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18573/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19977/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19304/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19288/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19793/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.19656/",#10
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.18700/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16703/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16396/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17952/"
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17682/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.17158/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16196/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.16359/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15400/",#15
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15822/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.15543/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14110/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14720/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.14961/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13597/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13568/",#20
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.13161/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.12213/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11622/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.12044/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11042/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.11751/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.09088/",
               "/home/data/jy/GLIP/OUTPUTcoco3s_2017_val/0.09521/",

               ]
    for image_id in os.listdir(submits[1]):

        allpic = [ "000000000669.jpg", "000000000721.jpg",]
        allpic = ["000000000669.jpg", ]
        if image_id not in allpic:
            continue
        # allpic = ["000000000186.jpg","000000000027.jpg",
        #           "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
        #           "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
        #           "000000000456.jpg","000000000023.jpg",]
        # allpic = [3183, 2535, 8496]
        import json
        f = open("/home/data/jy/GLIP/DATASET/coco3s/annotations/instances_val2017.json", 'r')
        cocogt_dataset = json.load(f)

        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        valdata_dir = '/home/data/jy/GLIP/DATASET/coco3s/val2017'
        # for comp in [1,2]:
        #     fp = open("{}/mask.json".format(submits[comp]), 'r')
        #     preds = json.load(fp)
        #     image = cv2.imread(valdata_dir + '/%s' % image_id)
        #     image[:,:,:] =0
        #     pred_im=image.copy()
        #     color=[0,255,255]
        #     for pred in cocogt_dataset['annotations']:
        #         if pred['image_id']==int(image_id[:-4]):
        #             segmentation=pred['segmentation']
        #             points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
        #             before_overlay = image.copy()
        #             # cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        #             cv2.fillPoly(image, [points], color=color)
        #             # x, y, w, h = pred['bbox']
        #             # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), [255,0,0], 2)
        #     mask_r=np.zeros((256,256))
        #     for pred_r in preds:
        #         if pred_r['image_id'] == int(image_id[:-4]):
        #             segmentation = pred_r['segmentation']  # 格式: [[x1,y1, x2,y2, ..., xn,yn]]
        #             mask = maskUtils.decode(segmentation)
        #             mask = mask.squeeze() if mask.ndim == 3 else mask
        #             mask_r+=mask
        #     pred_im[mask_r != 0] = [255,0,0]
        #     # pred_im[:,:,0][mask_r!=0]=0
        #     # pred_im[:, :, 1][mask_r != 0] = 255
        #     # pred_im[:, :, 2][mask_r != 0] = 0
        #     # plt.imshow(image);plt.show()
        #     # plt.imshow(pred_im);plt.show()
        #     add_up=image+pred_im
        #     out=change_color(add_up,[0,255,255],[220,220,0])
        #     plt.subplot(1,2,comp)
        #     plt.imshow(out)
        # plt.show()
        maski2submit={721:[1,5],669:[2,9]}
        for maski in [1,2]:
            fp = open("{}/mask.json".format(submits[maski2submit[int(image_id[:-4])][maski-1]]), 'r')
            preds = json.load(fp)
            image = cv2.imread(valdata_dir + '/%s' % image_id)
            imageori = cv2.imread(valdata_dir + '/%s' % image_id)
            CLASS_COLOR=[[0,0,0],[215,155,22],[244,12,100],[2,222,150],[111,19,187],[111,150,190]]
            image[:,:,:] =0
            image_maskimprompt=image.copy()
            image_moment = image.copy()
            pred_im=image.copy()
            pred_im_maskprompt = image.copy()
            pred_im_moment = image.copy()
            color=[0,255,255]
            mask_r = np.zeros((256, 256))
            for predid,pred in enumerate(cocogt_dataset['annotations']):
                if pred['image_id']==int(image_id[:-4]):
                    segmentation=pred['segmentation']
                    points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
                    before_overlay = image.copy()
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                    cv2.polylines(imageori, [points], isClosed=True, color=CLASS_COLOR[pred['category_id']], thickness=2)
                    cv2.fillPoly(image_maskimprompt, [points], color=color)
                    M=cv2.moments(points)
                    cx=int(M['m10']/M['m00'])
                    cy=int(M['m01']/M['m00'])
                    cv2.circle(image_moment,(cx,cy),3,color,-1)
                    cv2.circle(imageori, (cx, cy), 3, CLASS_COLOR[pred['category_id']], -1)
                    # x, y, w, h = pred['bbox']
                    # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), [255,0,0], 2)
            ranks = {721: [0,1, 2,3,4,5,6,7,8,9,10,12,15,17,18,23],
                     669: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,21,22,25,],}
            #       8496:[0,1,2,5,3,8]}
            k=-1
            for pred_rid,pred_r in enumerate(preds):
                if pred_r['image_id'] == int(image_id[:-4]):
                    k+=1
                    if k not in ranks[pred_r['image_id']] and maski==1:
                        continue
                    segmentation = pred_r['segmentation']  # 格式: [[x1,y1, x2,y2, ..., xn,yn]]
                    mask = maskUtils.decode(segmentation)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    # plt.imshow(mask);plt.show()
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    cv2.drawContours(pred_im, [cnt], 0, [255,0,0], 3)
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(pred_im_moment, (cx, cy), 3, [255, 0, 0], -1)
                    mask_r+=mask
            pred_im_maskprompt[mask_r != 0] = [255,0,0]
            # pred_im[:,:,0][mask_r!=0]=0
            # pred_im[:, :, 1][mask_r != 0] = 255
            # pred_im[:, :, 2][mask_r != 0] = 0
            # plt.imshow(image);plt.show()
            # plt.imshow(pred_im);plt.show()
            add_up=image+pred_im
            add_up_maskimprompt = image_maskimprompt + pred_im_maskprompt
            add_up_moment= image_moment+ pred_im_moment
            out=change_color(add_up,[0,255,255],[220,220,0])
            out_maskimprompt = change_color(add_up_maskimprompt, [0, 255, 255], [220, 220, 0])
            out_moment = change_color(add_up_moment, [0, 255, 255], [220, 220, 0])
            H=3;W=2
            plt.subplot(H,W,maski)
            plt.imshow(out_maskimprompt)
            plt.subplot(H,W,maski+2)
            plt.imshow(out)
            plt.subplot(H,W,maski+4)
            plt.imshow(out_moment)
            io.imsave('TOSHOW2025tmi2/{}_maskimprompt{}.png'.format(image_id,maski),out_maskimprompt)
            io.imsave('TOSHOW2025tmi2/{}_contour{}.png'.format(image_id,maski), out)
            io.imsave('TOSHOW2025tmi2/{}_moment{}.png'.format(image_id,maski), out_moment)
            io.imsave('TOSHOW2025tmi2/{}_GT.png'.format(image_id), imageori)
        plt.show()

        for maski in [1,2]:
            fp = open("{}/mask.json".format(submits[maski]), 'r')
            preds = json.load(fp)
            image = cv2.imread(valdata_dir + '/%s' % image_id)
            image[:,:,:] =0
            pred_im=image.copy()
            color=[0,255,255]
            for pred in cocogt_dataset['annotations']:
                if pred['image_id']==int(image_id[:-4]):
                    segmentation=pred['segmentation']
                    SINGLE_CELL_MAP=np.zeros((256,256,3))
                    points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
                    before_overlay = image.copy()
                    # cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                    # cv2.fillPoly(SINGLE_CELL_MAP, [points], color=color)
                    # contours, hierarchy = cv2.findContours(SINGLE_CELL_MAP, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # cnt = contours[0]
                    M=cv2.moments(points)
                    cx=int(M['m10']/M['m00'])
                    cy=int(M['m01']/M['m00'])
                    cv2.circle(image,(cx,cy),2,color,4)
                    # x, y, w, h = pred['bbox']
                    # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), [255,0,0], 2)
            mask_r=np.zeros((256,256))
            for pred_r in preds:
                if pred_r['image_id'] == int(image_id[:-4]):
                    segmentation = pred_r['segmentation']  # 格式: [[x1,y1, x2,y2, ..., xn,yn]]
                    mask = maskUtils.decode(segmentation)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(pred_im, (cx, cy), 2, [255,0,0], 4)
            #         mask_r+=mask
            # pred_im[mask_r != 0] = [255,0,0]
            # pred_im[:,:,0][mask_r!=0]=0
            # pred_im[:, :, 1][mask_r != 0] = 255
            # pred_im[:, :, 2][mask_r != 0] = 0
            # plt.imshow(image);plt.show()
            # plt.imshow(pred_im);plt.show()
            add_up=image+pred_im
            out=change_color(add_up,[0,255,255],[220,220,0])
            plt.subplot(1,2,maski)
            plt.imshow(out)
        plt.show()
        if image_id not in allpic:
            continue
        NEEDTOSHOW=False
        for picid in allpic:
            if '{}'.format(picid) in image_id:
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        # if not NEEDTOSHOW:
        #     continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 11
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 0.9*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz12345678901234567890'
        for iid in range(4*COL_NUMS):
            idx += 1
            plt.subplot(H, COL_NUMS, idx)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))

                # plt.title('({})'.format(iid), y=-0.1, color='k', fontsize=20)
                if iid==(4*COL_NUMS-2):
                    # plt.close()
                    im_method_GT = io.imread(submits[rank[0]] + '/' + image_id)[:, :, :]
                    im_method[210:255,50:150,:]=im_method_GT[210:255,50:150,:]
                    im_method_GT2 = io.imread(submits[rank[11]] + '/' + image_id)[:, :, :]
                    im_method[100:160,135:163,:]=im_method_GT2[100:160,135:163,:]
                    # plt.imshow(im_method);plt.show()
                    im_method[140:200, 161:220, :] = im_method_GT[140:200, 161:220, :]
                    # plt.imshow(im_method);plt.show()

                plt.imshow(im_method[:,:,:])
            except:
                pass
            if iid>=3*COL_NUMS:
                plt.title('({})'.format(TITLE[iid-3*COL_NUMS]),y=-0.168,color='k',fontsize=40)
            # print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW2025tmi2/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW2025tmi2/ALLcom.png')
    map=io.imread('TOSHOW2025tmi2/ALLcom.png')
    io.imsave('TOSHOW2025tmi2/ALLcom.png',map[:-300,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_COLORED_ALLSHOWWITH_OR_WITHOUT():
    fid = 0
    submits = ["/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/GT/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.22411/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.22305/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.22909/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.22034/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.19272/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.17930/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.18174/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15129/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.12143/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08656/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.28124/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.27039/",
                              "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.26471/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.24936/",#5
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.20576/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.26655/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.17041/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.16846/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15769/",#10
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15415/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10834/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08137/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10627/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08620/",#15
                              "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07695/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07728/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07802/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.07383/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.06003/",
               '/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.01511/',
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.12396/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.13447/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.15052/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.13447/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.14786/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.13349/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.12074/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.11317/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.16650/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.10099/",
               "/home/data/jy/GLIP/OUTPUTcoco1s_2017_val/0.08498/",
               ]
    for image_id in os.listdir(submits[1]):

        allpic = [ "000000000435.jpg", ]
        allpic = ["000000000515.jpg", ]
        if image_id not in allpic:
            continue
        import json
        f = open("/home/data/jy/GLIP/DATASET/coco1s/annotations/instances_train2017.json", 'r')
        cocogt_dataset = json.load(f)
        valdata_dir = '/home/data/jy/GLIP/DATASET/coco1s/train2017'
        maski2submit = {435: [1,4], 367: [1,4],515:[1,2]}
        for maski in [1, 2]:
            fp = open("{}/mask.json".format(submits[maski2submit[int(image_id[:-4])][maski - 1]]), 'r')
            preds = json.load(fp)
            image = cv2.imread(valdata_dir + '/%s.png' % image_id[:-4])
            imageori = cv2.imread(valdata_dir + '/%s.png' % image_id[:-4])
            CLASS_COLOR = [[0, 0, 0], [255, 155, 22], [254, 12, 55], [255, 212, 20], [11, 109, 255], [191, 255, 180]]
            image[:, :, :] = 0
            binary=image.copy()
            image_maskimprompt = image.copy()
            image_moment = image.copy()
            pred_im = image.copy()
            pred_im_maskprompt = image.copy()
            pred_im_moment = image.copy()
            color = [0, 255, 255]
            mask_r = np.zeros((256, 256))
            for predid, pred in enumerate(cocogt_dataset['annotations']):
                if pred['image_id'] == int(image_id[:-4]):
                    segmentation = pred['segmentation']
                    points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                    cv2.polylines(imageori, [points], isClosed=True, color=CLASS_COLOR[pred['category_id']],
                                  thickness=2)
                    cv2.fillPoly(image_maskimprompt, [points], color=color)
                    M = cv2.moments(points)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.fillPoly(binary, [points], color=[255,255,255])
                    cv2.circle(image_moment, (cx, cy), 3, color, -1)
                    cv2.circle(imageori, (cx, cy), 3, CLASS_COLOR[pred['category_id']], -1)
            io.imsave('515_binary.png',binary)
                    # x, y, w, h = pred['bbox']
                    # cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), [255,0,0], 2)
            ranks = {721: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 23],
                     669: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 25, ],
                        435:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,27],
                     367:[0,1,2,3,4,5,9,10,11,16,21]
                     }
            #       8496:[0,1,2,5,3,8]}
            k = -1
            for pred_rid, pred_r in enumerate(preds):
                if pred_r['image_id'] == int(image_id[:-4]):
                    k += 1
                    if k not in ranks[pred_r['image_id']] and maski == 1:
                        continue
                    segmentation = pred_r['segmentation']  # 格式: [[x1,y1, x2,y2, ..., xn,yn]]
                    mask = maskUtils.decode(segmentation)
                    mask = mask.squeeze() if mask.ndim == 3 else mask
                    # plt.imshow(mask);plt.show()
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    cv2.drawContours(pred_im, [cnt], 0, [255, 0, 0], 2)
                    M = cv2.moments(cnt)
                    try:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(pred_im_moment, (cx, cy), 3, [255, 0, 0], -1)
                    except:
                        pass
                    mask_r += mask
            # gtids=[11793,11795,11796]
            # for predid, pred in enumerate(cocogt_dataset['annotations']):
            #     if pred['image_id'] == int(image_id[:-4]) and predid in gtids:
            #         segmentation = pred['segmentation']
            #         points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))
            #         # image[:, :, :] = 0
            #         cv2.polylines(pred_im, [points], isClosed=True, color=[255,255,255], thickness=2)
            #         cv2.fillPoly(image_maskimprompt, [points], color=[255,255,255])
                    # plt.imshow(image);plt.show()
                    # pass
            pred_im_maskprompt[mask_r != 0] = [255, 0, 0]
            # pred_im[:,:,0][mask_r!=0]=0
            # pred_im[:, :, 1][mask_r != 0] = 255
            # pred_im[:, :, 2][mask_r != 0] = 0
            # plt.imshow(image);plt.show()
            # plt.imshow(pred_im);plt.show()
            add_up = image + pred_im
            add_up_maskimprompt = image_maskimprompt + pred_im_maskprompt
            add_up_moment = image_moment + pred_im_moment
            out = change_color(add_up, [0, 255, 255], [220, 220, 0])
            out_maskimprompt = change_color(add_up_maskimprompt, [0, 255, 255], [220, 220, 0])
            out_moment = change_color(add_up_moment, [0, 255, 255], [220, 220, 0])
            H = 3;
            W = 2
            plt.subplot(H, W, maski)
            plt.imshow(out_maskimprompt)
            plt.subplot(H, W, maski + 2)
            plt.imshow(out)
            plt.subplot(H, W, maski + 4)
            plt.imshow(out_moment)
            io.imsave('TOSHOW2025tmi/{}_maskimprompt{}.png'.format(image_id, maski), out_maskimprompt)
            io.imsave('TOSHOW2025tmi/{}_contour{}.png'.format(image_id, maski), out)
            io.imsave('TOSHOW2025tmi/{}_moment{}.png'.format(image_id, maski), out_moment)
            io.imsave('TOSHOW2025tmi/{}_GT.png'.format(image_id), imageori)
        plt.show()

        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 11
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 0.9*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz12345678901234567890'
        for iid in range(4*COL_NUMS):
            idx += 1
            plt.subplot(H, COL_NUMS, idx)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            try:
                # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))

                # plt.title('({})'.format(iid), y=-0.1, color='k', fontsize=20)
                if iid==(4*COL_NUMS-2):
                    # plt.close()
                    im_method_GT = io.imread(submits[rank[0]] + '/' + image_id)[:, :, :]
                    im_method[210:255,50:150,:]=im_method_GT[210:255,50:150,:]
                    im_method_GT2 = io.imread(submits[rank[11]] + '/' + image_id)[:, :, :]
                    im_method[100:160,135:163,:]=im_method_GT2[100:160,135:163,:]
                    # plt.imshow(im_method);plt.show()
                    im_method[140:200, 161:220, :] = im_method_GT[140:200, 161:220, :]
                    # plt.imshow(im_method);plt.show()

                plt.imshow(im_method[:,:,:])
            except:
                pass
            if iid>=3*COL_NUMS:
                plt.title('({})'.format(TITLE[iid-3*COL_NUMS]),y=-0.168,color='k',fontsize=40)
            # print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW2025tmi2/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW2025tmi2/ALLcom.png')
    map=io.imread('TOSHOW2025tmi2/ALLcom.png')
    io.imsave('TOSHOW2025tmi2/ALLcom.png',map[:-300,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_WITHORWITHOUT():
    allpic=["/home/data/jy/GLIP/TOSHOW2025tmi/000000000435.jpg_GT.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000435.jpg_maskimprompt1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000435.jpg_maskimprompt20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000435.jpg_contour1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000435.jpg_contour20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000435.jpg_moment1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000435.jpg_moment20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000367.jpg_GT0.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000367.jpg_maskimprompt10.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000367.jpg_maskimprompt20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000367.jpg_contour10.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000367.jpg_contour20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000367.jpg_moment1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi/000000000367.jpg_moment20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000669.jpg_GT0.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000669.jpg_maskimprompt1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000669.jpg_maskimprompt20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000669.jpg_contour1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000669.jpg_contour20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000669.jpg_moment1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000669.jpg_moment20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000721.jpg_GT0.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000721.jpg_maskimprompt1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000721.jpg_maskimprompt20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000721.jpg_contour1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000721.jpg_contour20.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000721.jpg_moment1.png",
            "/home/data/jy/GLIP/TOSHOW2025tmi2/000000000721.jpg_moment20.png",

            ]
    H = 5
    COL_NUMS = 7
    INCH = 20
    fig = plt.gcf()
    fig.set_size_inches(1.7 * INCH, 1.2 * INCH)
    TITLE = 'abcdefghijklmnopqrstuvwxyz12345678901234567890'
    for iid,impath in enumerate(allpic):
        plt.subplot(H, COL_NUMS, iid+1)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        try:
            # im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
            im_method = io.imread(impath)
            # if fid<3 and iid==0:
            #     im_method=transform.resize(im_method,(1088,800,3))
            # elif fid==3 :#and iid==0:
            #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))

            # plt.title('({})'.format(iid), y=-0.1, color='k', fontsize=20)


            plt.imshow(im_method[:, :, :])
        except:
            pass
        if iid >= 3 * COL_NUMS:
            plt.title('({})'.format(TITLE[iid - 3 * COL_NUMS]), y=-0.165, color='k', fontsize=40)
        # print(TITLE[iid])
    # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.title(image_id[:20]+'.jpg',y=0)
    plt.margins(0, 0)
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)


    plt.savefig('TOSHOW2025tmi/ALLWO.png')
    map = io.imread('TOSHOW2025tmi/ALLWO.png')
    io.imsave('TOSHOW2025tmi/ALLWO.png', map[:-370, :, :])


    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def plot_tsne():
    from openTSNE import TSNE
    # from examples import utils
    import numpy as np
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import gzip
    import pickle

    with gzip.open("macosko_2015.pkl.gz", "rb") as f:
        data = pickle.load(f)
    x = data["pca_50"]
    y = data["CellType1"].astype(str)
    print("Data set contains %d samples with %d features" % x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
    print("%d training samples" % x_train.shape[0])
    print("%d test samples" % x_test.shape[0])
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    embedding_train = tsne.fit(x_train)
# print_CVPR2025_ALLSHOW_COMPARE()
# print_CVPR2025_ALLSHOW_COMPARE_FINALMAP_afterselect()
# print_CVPR2025_ALLSHOW_COMPARE_rebuttal()
# print_CVPR2025_CAM()
# print_CVPR2025_CAMFINAL_RE()
# print_CVPR2025_CAM148()
# print_CVPR2025_VOCSPLITX3()
# print_CVPR2025_MEDICALX3()
# print_CVPR2025_ODINW13_final()
# print_CVPR2025_MONU()
# print_tasks_PLOT5comb(i=5,fid=9)#hole
# print_tasks_PLOT7comb()
# print_tasks_PLOT8comb()
# print_self_training_suplvisPLOT_COMB()
# print_self_training_supVOCPLOT2_COMB()
# print_CVPR2025_CAMFINAL_iccvrebuttal()
print_self_training_COLORED_ALLSHOWWITH_OR_WITHOUT()
print_WITHORWITHOUT()