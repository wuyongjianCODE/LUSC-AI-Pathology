import matplotlib.pyplot as plt
import os
from skimage import io,transform
def print_ALL_with_metreics_withCOMPARE():
    fid=0
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='.._.._logs_RESULTIMGS_LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4'
    submits=['/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 22:00:26_382902/',#GT
        "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_scn_continue/val2023-03-09 20:05:52_150623/",#22
           '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sc_412/val2023-03-09 20:14:07_328440/',#35
             '/data1/wyj/M/samples/PRM/YOLOX/val_vlplm/',#33
           '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:06:29_049792/',# '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_scn_275/val2023-03-09 19:02:29_255141/',#17
           # '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sc_412/val2023-03-09 20:27:55_874946/',#413
           # '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sn/val2023-03-09 21:20:14_418623/',#336
           "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 21:51:22_686128/",#414
             '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/fullsup/',#full44

    ]
    INCH = 20
    H = 16
    fig = plt.gcf()
    fig.set_size_inches(INCH, 2 * INCH)
    HAS_=0
    COL_NUMS=8
    for image_id in range(480):
        try:
            oriim=io.imread('datasets/COCO/val2017/%012d.jpg'%(image_id))
        except:
            continue
        allpic = [17,178,231,273,282]
        allpic = [178, 231]
            #print(dices)
    # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
           # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        if image_id not in allpic:
            continue
        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 1
        for submit in submits:
            idx += 1
            plt.subplot(H, COL_NUMS, idx + (fid%H) * COL_NUMS)
            plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.05,wspace=0.05,hspace=0.05)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            im_method=io.imread(submit+'/%012d.jpg'%(image_id))
            plt.imshow(im_method)
        plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(oriim)
        # plt.title('datasets/COCO/val2017/%012d.jpg'%(image_id),y=-0.15)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    TITLE='abcdefg'
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training0():
    fid=0
    submits = ['/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 22:00:26_382902/',  # GT
                '/data1/wyj/M/samples/PRM/YOLOX/val_sc_json/',#sc
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_30_ckpt', #'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_40_ckpt',  # 359
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_70_ckpt',  # 404
                "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 21:51:22_686128/",  # 414
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_150_ckpt',  # 414
                ]
    INCH = 20
    H = 16
    fig = plt.gcf()
    fig.set_size_inches(INCH, 2 * INCH)
    HAS_=0
    COL_NUMS=8
    for image_id in range(480):
        try:
            oriim=io.imread('datasets/COCO/val2017/%012d.jpg'%(image_id))
        except:
            continue

        allpic = [178, 231]
        allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        allpic = [204,]
        if image_id not in allpic:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 1
        for submit in submits:
            idx += 1
            plt.subplot(H, COL_NUMS, idx + (fid%H) * COL_NUMS)
            plt.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.025,wspace=0.05,hspace=0.05)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            im_method=io.imread(submit+'/%012d.jpg'%(image_id))
            plt.imshow(im_method)
        plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(oriim)
        # plt.title('datasets/COCO/val2017/%012d.jpg'%(image_id),y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    TITLE='abcdefg'
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
def print_self_training():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
               '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
                '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
             '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
                '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
               '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
               '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
               '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
                ]
    for image_id in os.listdir('ORI_WITH_BOX')[16:]:
        allpic = [3120,8317,2367,3183,8348,2535,8496,3150,2586]
        allpic = [3183, 2535, 8496]
        ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
              8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
              8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic:
            if 'IMG_{}'.format(picid) in image_id:
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 4
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.5*INCH, 1.33*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid,submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                if fid<3 and iid==0:
                    im_method=transform.resize(im_method,(1088,800,3))
                elif fid==3 :#and iid==0:
                    im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:-10,:,:])
            except:
                pass
            if fid == 3:
                plt.title('({})'.format(TITLE[iid]),y=-0.1,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    map=io.imread('TOSHOW/ALLcom.png')
    io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_sup():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    for image_id in os.listdir('ORI_WITH_BOX')[:]:
        allpic = ['000000020107','000000008629','000000010092','000000013597','000000026465',
                  '000000003255','000000025560','000000031248','000000023034','000000016439',
                  '000000029397','000000021167','000000015254','000000009769','000000025139'
                  ]
        allpic_batch2=[
                  '000000029397','000000021167','000000015254','000000009769',#'000000025139',
                  ]
        allpic_batch2=[
                  '000000020107','000000013597','000000023034','000000009769',
                  ]
        ranks={allpic[0]:[0,5,6,7,8,2],allpic[1]:[0,2,5,7,8,6],allpic[2]:[0,7,4,5,8,3],allpic[3]:[0,1,3,7,8,6,],allpic[4]:[0,8,1,7,11,16],
               allpic[5]:[0,7,9,15,10,13],allpic[6]:[0,7,9,10,15,1],allpic[7]:[0,4,6,7,10,9],allpic[8]:[0,8,9,7,12,5],allpic[9]:[0,7,10,12,14,1],
               allpic[10]: [0,8,6,14,16,3],allpic[11]: [0,8,16,12,9,1],allpic[12]: [0,4,9,15,16,3],allpic[13]: [0,8,12,14,6,1],allpic[14]: [0,8,4,5,16,3],
               }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid,submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                print(submits[rank[iid]]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomVOC1.png')
    map=io.imread('TOSHOW/ALLcomVOC1.png')
    io.imsave('TOSHOW/ALLcomVOC1.png',map[:-200,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_supVOCPLOT():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000041_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0066495_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000082_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0053196_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000123_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0039897_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000164_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalvpt0_5_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000205_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_2imada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        # allpic = ['2007_0005490','2008_003430','2009_0001873','2008_003447','2010_003390',
        #           '2009_004118','2013_001019','2012_000176',
        #
        #           ]
        # allpic_batch2=[
        #           '2007_000549','000000021167','000000015254','000000009769',#'000000025139',
        #           ]
        # allpic_batch2=[
        #           '000000025560','000000029397','000000021167','000000015254',
        #           ]
        # ranks={allpic[0]:[1,5,6,7,8,9,15,28,21,4],allpic[1]:[1,6,7,8,30,22,28,14,5],allpic[2]:[1,34,36,35,19,23,25,9,5,20],allpic[3]:[1,31,33,38,23,24,19,3,34,5],allpic[4]:[1,6,8,20,25,26,27,33,22,30],
        #        allpic[5]:[1,20,21,27,29,9,10,2,34,3],allpic[6]:[1,41,36,40,34,32,28,2,3,9],allpic[7]:[1,39,41,37,31,26,23,18,4,34],allpic[8]:[1,8,10,17,24,26,28,46,2,3]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=True
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
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
        fig.set_size_inches(2*INCH,INCH)#len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[iid]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
                plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[iid]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCVVOC{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCVVOC.png')
    # map=io.imread('TOSHOW/ALLcomlv.png')
    # io.imsave('TOSHOW/ALLcomlv.png',map[:-250,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_supVOCPLOT2():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000041_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0066495_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000082_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0053196_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000123_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0039897_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000164_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalvpt0_5_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000205_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_2imada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['2010_004008','2008_008357','2007_003580','2010_002055','2008_004559',
                  '2008_006578',
                  ]
        allpic_batch2=allpic
        ranks={allpic[0]:[1,40,39,38,33,34,23,9,4,5],allpic[1]:[1,35,36,37,38,8,9,10,3,4],allpic[2]:[1,37,38,39,40,41,30,29,7,5],allpic[3]:[1,14,19,20,21,22,23,10,4,5],allpic[4]:[1,37,39,40,41,36,34,35,38,4],
               allpic[5]:[1,39,40,41,35,27,6,14,29,38],}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 7
        if fid % H == 0:
            plt.close()
        COL_NUMS = 10#len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.8*INCH,INCH)#len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits[:10]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]-1]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                im_method = transform.resize(im_method, (480, 600, 3))
                plt.imshow(im_method[:,:,:])

                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCVVOC{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCVVOC2.png')
    map=io.imread('TOSHOW/ALLcomECCVVOC2.png')
    io.imsave('TOSHOW/ALLcomECCVVOC2.png',map[:-220,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_supVOCPLOT2_COMB():
    fid=6
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000041_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0066495_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000082_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0053196_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000123_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalima_ft_task_1_model_0039897_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000164_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalvpt0_5_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_imada8_ft_task_1_model_0000205_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_2imada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['2009_000562','2008_003430','2009_000341','2008_003347','2010_003390',
                  '2009_004118','2011_001019','2011_000176','2012_1111111',

                  ]
        allpic_batch2=['2009_000562','2008_003430','2009_000341','2008_003347','2010_003390',
                 '2011_000176',

                  ]
        ranks={allpic[0]:[1,41,40,39,38,34,29,28,4,5],allpic[1]:[1,6,7,8,33,30,22,28,5,14],allpic[2]:[1,38,39,36,41,19,20,2,4,34],allpic[3]:[1,31,33,38,23,24,19,3,34,5],allpic[4]:[1,6,8,20,25,26,27,33,22,30],
               allpic[5]:[1,20,21,27,29,9,10,2,34,3],allpic[6]:[1,41,36,40,34,32,28,2,3,9],allpic[7]:[1,39,41,37,31,26,23,18,4,34],allpic[8]:[1,8,10,17,24,26,28,46,2,3]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 13
        # if fid % H == 0:
        #     plt.close()
        COL_NUMS = 10#len(submits)
        # INCH = 20
        # fig = plt.gcf()
        # fig.set_size_inches(1.8*INCH,INCH)#len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits[:10]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[rank[iid]-1]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                im_method = transform.resize(im_method, (480, 600, 3))
                plt.imshow(im_method[:,:,:])

                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    map=io.imread('TOSHOW/ALLcomECCV.png')
    io.imsave('TOSHOW/ALLcomECCV.png',map[:-220,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_suplvis():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",  #1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",  #2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",  #3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",  #4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",  #5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",  #6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",  #7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",  #8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",  #16
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",  #17
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",  #18
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",  #19
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",  #20
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",  #21
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",  #22
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",  #23
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",  #24
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",  #25
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",  #26
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",  #27
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",  #28
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",  #29
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",  #30
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",  #31
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",  #32
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",  #33
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",  #34
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",  #35
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",  #36
               "/home/data/jy/GLIP/PLOT/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",  #38
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               # 37


               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['000000010583','000000025986','000000018575','000000022479','000000022892',
                  '000000029187','000000007108','000000015272','000000026941','000000016439',
                  '000000027932','000000022935','000000017714','000000029397','000000012062'
                  ]
        allpic_batch2=[
                  '000000029397','000000021167','000000015254','000000009769',#'000000025139',
                  ]
        allpic_batch2=[
                  '000000025560','000000029397','000000021167','000000015254',
                  ]
        ranks={allpic[0]:[1,5,6,7,8,9,15,28,21,4],allpic[1]:[1,7,8,9,10,21,20,51,2,4],allpic[2]:[1,7,9,10,13,17,21,31,2,4],allpic[3]:[1,8,9,10,14,24,34,46,51,2,3],allpic[4]:[1,9,11,12,16,45,51,7,3,47],
               allpic[5]:[1,7,8,14,20,28,50,46,3,4],allpic[6]:[1,9,3,8,21,23,50,28,10,14],allpic[7]:[1,10,13,28,22,24,28,4,2,3],allpic[8]:[1,8,10,17,24,26,28,46,2,3],allpic[9]:[0,11,15,20,28,37,46,4,2,3],
               allpic[10]: [1,39,28,23,22,20,17,7,46,3],allpic[11]: [1,14,15,18,19,31,34,38,50,51],allpic[12]: [1,49,43,39,29,21,13,28,3,4],allpic[13]: [1,7,5,17,21,23,45,46,3,4],allpic[14]: [1,4,5,7,9,10,14,3,44,8],
               }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=True
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
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
        fig.set_size_inches(len(submits)*INCH/H,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        for iid,submit in enumerate(submits):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[iid]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
                plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            except:
                print(submits[iid]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.2,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    # map=io.imread('TOSHOW/ALLcomlv.png')
    # io.imsave('TOSHOW/ALLcomlv.png',map[:-250,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_suplvisPLOT():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",  #1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",  #2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",  #3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",  #4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",  #5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",  #6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",  #7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",  #8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",  #16
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",  #17
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",  #18
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",  #19
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",  #20
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",  #21
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",  #22
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",  #23
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",  #24
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",  #25
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",  #26
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",  #27
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",  #28
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",  #29
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",  #30
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",  #31
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",  #32
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",  #33
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",  #34
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",  #35
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",  #36
               "/home/data/jy/GLIP/PLOT/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",  #38
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               # 37


               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['000000010583','000000025986','000000018575','000000022479','000000022892',
                  '000000029187','000000007108','000000015272','000000026941','000000002592',
                  '000000027932','000000022935','000000017714','000000029397','000000012062'
                  ]
        allpic_batch2=[
                  '000000012062','000000029397','000000017714','000000029187','000000018575','000000026941',
                  ]
        # allpic_batch2=[
        #           '000000025560','000000029397','000000021167','000000015254',
        #           ]
        ranks={allpic[0]:[1,5,6,7,8,9,28,15,21,4],allpic[1]:[1,7,8,9,10,21,24,51,2,4],allpic[2]:[1,7,9,10,13,17,21,31,2,4],allpic[3]:[1,8,9,10,14,34,46,51,2,3],allpic[4]:[1,12,11,16,9,45,51,7,3,47],
               allpic[5]:[1,7,8,14,20,28,50,46,3,4],allpic[6]:[1,9,3,8,21,23,50,28,10,14],allpic[7]:[1,10,13,28,22,24,28,4,2,3],allpic[8]:[1,8,10,17,24,26,28,46,2,3],allpic[9]:[1,37,11,20,15,28,46,4,2,3],
               allpic[10]: [1,28,39,23,22,20,17,46,7,3],allpic[11]: [1,14,15,18,19,31,34,38,50,51],allpic[12]: [1,49,43,39,29,21,13,28,3,4],allpic[13]: [1,7,5,17,21,23,45,46,3,4],allpic[14]: [1,4,5,7,9,10,14,3,44,8],
               }
        rerank=[0,1,2,3,6,7,4,5,8,9]
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 7
        if fid % H == 0:
            plt.close()
        COL_NUMS = 10#len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.8*INCH,1*INCH)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        ONE_TIME_H=0
        for iid,submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            # try:
            im_method=io.imread(submits[rank[rerank[iid]]-1]+'/'+image_id)[:,:,:]
            h,w,c=im_method.shape
            # if fid<3 and iid==0:
            #     im_method=transform.resize(im_method,(1088,800,3))
            # elif fid==3 :#and iid==0:
            #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
            if ONE_TIME_H==0:
                ONE_TIME_H=h*(800/w)
            im_method = transform.resize(im_method, (480,600, 3))
            plt.imshow(im_method[:,:,:])
                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            # except:
            #     print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    map=io.imread('TOSHOW/ALLcomECCV.png')
    io.imsave('TOSHOW/ALLcomECCV.png',map[:-220,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_suplvisPLOT_COMB():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",  #1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",  #2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",  #3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",  #4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",  #5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",  #6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",  #7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",  #8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",  #16
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",  #17
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",  #18
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",  #19
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",  #20
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",  #21
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",  #22
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",  #23
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",  #24
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",  #25
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",  #26
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
               # "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",  #27
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",  #28
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",  #29
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",  #30
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",  #31
               "/home/data/jy/GLIP/PLOT/PLOT2/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",  #32
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0005000_pth/",  #33
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0010000_pth/",  #34
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0015000_pth/",  #35
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_bitcross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0005000_pth/",  #36
               "/home/data/jy/GLIP/PLOT/glip_large_model_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0030000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0025000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0020000_pth/",
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0015000_pth/",  #38
               "/home/data/jy/GLIP/PLOT/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_loracross_model_0010000_pth/",
               # 37


               ]
    BATCH_NAMES=[
                  '000000021167.jpg','000000029397.jpg','000000015254.jpg','000000025560.jpg',
                  ]
    for image_id in os.listdir(submits[-1]):
        allpic = ['000000010583','000000025986','000000018575','000000022479','000000022892',
                  '000000029187','000000007108','000000015272','000000026941','000000002592',
                  '000000027932','000000022935','000000017714','000000029397','000000012062'
                  ]
        allpic_batch2=[
                  '000000012062','000000029397','000000017714','000000029187','000000018575','000000026941',
                  ]
        # allpic_batch2=[
        #           '000000025560','000000029397','000000021167','000000015254',
        #           ]
        ranks={allpic[0]:[1,5,6,7,8,9,28,15,21,4],allpic[1]:[1,7,8,9,10,21,24,51,2,4],allpic[2]:[1,7,9,10,13,17,21,31,2,4],allpic[3]:[1,8,9,10,14,34,46,51,2,3],allpic[4]:[1,12,11,16,9,45,51,7,3,47],
               allpic[5]:[1,7,8,14,20,28,50,46,3,4],allpic[6]:[1,9,3,8,21,23,50,28,10,14],allpic[7]:[1,10,13,28,22,24,28,4,2,3],allpic[8]:[1,8,10,17,24,26,28,46,2,3],allpic[9]:[1,37,11,20,15,28,46,4,2,3],
               allpic[10]: [1,28,39,23,22,20,17,46,7,3],allpic[11]: [1,14,15,18,19,31,34,38,50,51],allpic[12]: [1,49,43,39,29,21,13,28,3,4],allpic[13]: [1,7,5,17,21,23,45,46,3,4],allpic[14]: [1,4,5,7,9,10,14,3,44,8],
               }
        rerank=[0,1,2,3,6,7,4,5,8,9]
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 13
        if fid % H == 0:
            plt.close()
        COL_NUMS = 10#len(submits)
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.8*INCH,1*INCH*13/7)#1.45*INCH, 0.8*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890123456789012345678901234567890'
        ONE_TIME_H=0
        for iid,submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            # try:
            im_method=io.imread(submits[rank[rerank[iid]]-1]+'/'+image_id)[:,:,:]
            h,w,c=im_method.shape
            # if fid<3 and iid==0:
            #     im_method=transform.resize(im_method,(1088,800,3))
            # elif fid==3 :#and iid==0:
            #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
            if ONE_TIME_H==0:
                ONE_TIME_H=h*(800/w)
            im_method = transform.resize(im_method, (480,600, 3))
            plt.imshow(im_method[:,:,:])
                # plt.title('({})'.format(idx), y=-0.2, color='k', fontsize=40)
            # except:
            #     print(submits[rank[iid]-1]+'/'+image_id)
            if fid == H-1:
                # plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid % H == 0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.savefig('TOSHOW/ALLcomECCV.png')
    # map=io.imread('TOSHOW/ALLcomECCV.png')
    # io.imsave('TOSHOW/ALLcomECCV.png',map[:-220,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_supVOC():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",#1
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",#2
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",#3
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",#4
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",#5
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",#6
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",#7
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",#8
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    for image_id in os.listdir('ORI_WITH_BOX')[:]:
        allpic = ['000000020107','000000008629','000000010092','000000013597','000000026465',
                  '000000003255','000000025560','000000031248','000000023034','000000016439',
                  '000000029397','000000021167','000000015254','000000009769','000000025139',
                  '2008_003430','2009_000562','2007_006260','2011_000882','2011_002031',
                  '2008_007433','2011_003545','2009_000142','2007_009527',
                  ]
        allpic_batch2=[
            '2008_003430', '2009_000562', '2007_006260', '2011_000882', '2011_002031',
            '2008_007433', '2011_003545', '2009_000142', '2007_009527',
                  ]
        allpic_batch2=[
            '2008_003430','2011_003545', '2009_000142', '2007_009527',
                  ]
        ranks={allpic[0]:[0,5,6,7,8,2],allpic[1]:[0,2,5,7,8,6],allpic[2]:[0,7,4,5,8,3],allpic[3]:[0,1,3,7,8,6,],allpic[4]:[0,8,1,7,11,16],
               allpic[5]:[0,7,9,15,10,13],allpic[6]:[0,7,9,10,15,1],allpic[7]:[0,4,6,7,10,9],allpic[8]:[0,8,9,7,12,5],allpic[9]:[0,7,10,12,14,1],
               allpic[10]: [0,8,6,14,16,3],allpic[11]: [0,8,16,12,9,1],allpic[12]: [0,4,9,15,16,3],allpic[13]: [0,8,12,14,6,1],allpic[14]: [0,8,4,5,16,3],
               allpic[15]: [0, 7,6,5,4,3],allpic[16]: [0,6,7,8,12,3],allpic[17]: [0,16,6,8,3,1],allpic[18]: [0,3,12,8,6,1],allpic[19]: [0,7,8,12,16,1],
               allpic[20]: [0, 1,16,4,7,8],allpic[21]: [0,16,5,8,4,1],allpic[22]: [0,16,6,7,8,1],allpic[23]: [0,7,8,16,12,1],
               }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

        NEEDTOSHOW=False
        for picid in allpic_batch2:
            if picid in image_id:
                print(picid)
                NEEDTOSHOW=True
                SHORTID=picid
                rank=ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.47*INCH, 0.9*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid,submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.01,hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            try:
                im_method=io.imread(submits[iid]+'/'+image_id)[:,:,:]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:,:,:])
            except:
                print(submits[rank[iid]]+'/'+image_id)
            if fid == H-1:
                plt.title('({})'.format(TITLE[iid]),y=-0.21,color='k',fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcomECCV{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomECCV.png')
    map=io.imread('TOSHOW/ALLcomECCV.png')
    io.imsave('TOSHOW/ALLcomECCV.png',map[:-400,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
# submits = ['ORI_WITH_BOX',  # GT
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
#          '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
#             ]
# im_method=io.imread(submits[4]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method2=io.imread(submits[5]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method[500:,:,:]=im_method2[500:,:,:]
# io.imsave('/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg',im_method)
def print_self_training_ALLSHOW():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "OUTPUT1/SSNS",#1
               "OUTPUT1/ORI0.26811",#2
               "OUTPUT1/ORI0.25247",#3
               "OUTPUT1/ORI0.30384",#4
               "OUTPUT1/SOP1",#5
               "OUTPUT1/ORI0.11968",#PSM6
               "OUTPUT1/ORI0.18027",#cutler7
               "OUTPUT1/ORI0.33063",#vlplm8
               "OUTPUT1/VLDET",#9
               "OUTPUT1/ORI0.22036",#10
               "OUTPUT1/MICCAI",
               "OUTPUT1/fullsup",
               "OUTPUT1/OURS",#"OUTPUT1/ORI0.42482",
               "OUTPUT1/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0
        if image_id not in allpic:
            continue
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
        TITLE = 'abcdefghijklmnopq'
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
            plt.savefig('TOSHOWtmi/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
# submits = ['ORI_WITH_BOX',  # GT
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
#          '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
#             ]
# im_method=io.imread(submits[4]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method2=io.imread(submits[5]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method[500:,:,:]=im_method2[500:,:,:]
# io.imsave('/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg',im_method)
def print_self_training_ALLSHOW_CONSEP():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "OUTPUT2/0.25036",#1
               "OUTPUT2/0.17738",#2
               "OUTPUT2/0.19671",#3
               "OUTPUT2/0.22541",#4
               "OUTPUT2/ORI0.22912",#5
               "OUTPUT2/ORI0.12272",#PSM6
               "OUTPUT2/ORI0.09627",#cutler7
               "OUTPUT2/0.15668",#vlplm8
               "OUTPUT2/0.11232",#9
               "OUTPUT2/0.13487",#10
               "OUTPUT2/0.33675",
               "OUTPUT2/0.36208",
               "OUTPUT2/0.38245",#"OUTPUT1/ORI0.42482",
               "OUTPUT2/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        # allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
        #           "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
        #           "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
        #           "000000000458.jpg","000000000460.jpg",]
        # if image_id not in allpic:
        #     continue
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

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
        TITLE = 'abcdefghijklmnopq'
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
            plt.savefig('TOSHOWtmi/ALLcom{}_consep.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom_consep.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_ALLSHOW_VOC():
    fid=0
    submits = ['ORI_WITH_BOX',  # GT
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0013299_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_pascalla_ft_task_1_model_0026598_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_4_model_0007131_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_2_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_3_model_best_pth/",
               "/home/data/jy/GLIP/PLOT2/_home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_laada8_ft_task_1_model_best_pth/",
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",  # 9
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",  # 10
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",  # 11
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",  # 12
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",  # 13
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",  # 14
               "/home/data/jy/GLIP/PLOT/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",  # 15
               "/home/data/jy/GLIP/PLOT/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",  # 16
               ]
    for image_id in os.listdir('ORI_WITH_BOX'):
        if '000000' in image_id:
            continue
        # allpic = [3120,8317,2367,3183,8348,2535,8496,3150,2586]
        # allpic = [3183, 2535, 8496]
        # ranks={3120:[0,3,1,6,4,2],8317:[0,3,2,5,4,1],2367:[0,3,4,5,1,2],3183:[0,1,5,5,2,4],8348:[0,3,2,5,6,4],2535:[0,4,2,3,1,5],
        #       8496:[0,1,2,2,5,4],3150:[0,5,1,3,4,2],2586:[0,1,2,3,4,5]}
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID=0

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
        H = 18
        if fid % H == 0:
            plt.close()
        COL_NUMS = 17
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2*INCH, 2*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopq'
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
            plt.savefig('TOSHOW/ALLcomVOC{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcomVOC.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
# submits = ['ORI_WITH_BOX',  # GT
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_glip_large_model.pth/',  # 359,1
#            '/data2/wyj/GLIP/_data2_wyj_GLIP_OUTPUT_TRAIN_coco_ta_model_0010000.pth/',#2
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/',#3'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
#          '/data2/wyj/GLIP/_data2_wyj_GLIP_COCO_model_0045000_0569.pth/',#4
#             '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/',#5
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001300.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/',
#            '/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0040000.pth/',
#             ]
# im_method=io.imread(submits[4]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method2=io.imread(submits[5]+'/'+'IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg')
# im_method[500:,:,:]=im_method2[500:,:,:]
# io.imsave('/data2/wyj/GLIP/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest2_model_0020000.pth/IMG_8496_MOV-2_jpg.rf.a07bf41c4d61ee18223b3cf390c8fdda.jpg',im_method)
import numpy as np
import numpy as np
def is_bbox_color(color):
    # 判断像素颜色是否满足边框颜色条件
    return color[0] >240 and color[2] >240 and color[1] > 200

def find_bboxes(image):
    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 创建一个空列表来存储所有bbox的坐标
    bboxes = []

    # 遍历图像的每个像素
    for y in range(height):
        for x in range(width):
            # 检查当前像素是否在图像边界内
            if x < width and y < height:
                # 获取当前像素的颜色值
                color = image[y, x]

                # 检查当前像素是否为边框颜色
                if is_bbox_color(color):
                    # 找到了边框颜色，开始扩展bbox

                    # 初始化bbox的左上和右下坐标
                    top_left = (x, y)
                    bottom_right = (x, y)

                    # 扩展bbox的右下坐标
                    while x < width and is_bbox_color(image[y, x]):
                        bottom_right = (x, y)
                        x += 1

                    # 扩展bbox的右下坐标
                    while y < height and is_bbox_color(image[y, bottom_right[0]]):
                        bottom_right = (bottom_right[0], y)
                        y += 1

                    # 将bbox的左上和右下坐标添加到列表中
                    bboxes.append((top_left, bottom_right))

    # 返回所有bbox的坐标列表
    return bboxes
def print_self_training_COLORED_ALLSHOW():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "/home/data/jy/GLIP/OUTPUT1/0.30131/",#1ssns 354
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.2938111707351383/",#2 275
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.27047253330669924/",#3  262
               "/home/data/jy/GLIP/OUTPUT1/0.25163/",#4 292
               "/home/data/jy/GLIP/OUTPUT1/0.19769/",#5 235
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.22818376550457384/",#PSM6 227
               "/home/data/jy/GLIP/OUTPUT1/0.11955/",#cutler7 115
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.2153719550268572/",#vlplm8 333
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.17205429054974897/",#9 173
               "/home/data/jy/GLIP/COMP_OUTPUTS/0.08508564588325737/",#10 07
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.3984633510574706/",#416
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.4165950561751854/",  #425
               # "/home/data/jy/GLIP/OUTPUT1/0.47323/",
        "/home/data/jy/GLIP/COMP_OUTPUTS/0.4148790568704443/",
        "/home/data/jy/GLIP/OUTPUT1/0.44415/",
        "/home/data/jy/GLIP/COMP_OUTPUTS/0.41517412385919217/",

               # "OUTPUT1/GT",  # GT
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
        if image_id not in allpic:
            continue
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
        TITLE = 'abcdefghijklmnopq'
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
            plt.savefig('TOSHOWtmi/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training_COLORED_ALLSHOW_CONSEP():
    fid=0
    submits = [#'/home/data/jy/GLIP/DATASET/coco1/val2017/',  # ori
               "/home/data/jy/GLIP/OUTPUT2/0.27752/",#1ssns 354
               "/home/data/jy/GLIP/OUTPUT2/0.22034/",#2 275
               "/home/data/jy/GLIP/OUTPUT2/0.15932/",#3  262
               "/home/data/jy/GLIP/OUTPUT2/0.23618/",#4 292
               "/home/data/jy/GLIP/OUTPUT2/0.12168/",#5 235
               "/home/data/jy/GLIP/OUTPUT2/0.21598/",#PSM6 227
               "/home/data/jy/GLIP/OUTPUT2/0.10464/",#cutler7 115
               "/home/data/jy/GLIP/OUTPUT2/0.19242/",#vlplm8 333
               "/home/data/jy/GLIP/OUTPUT2/0.12571/",#9 173
               "/home/data/jy/GLIP/OUTPUT2/0.04011/",#"/home/data/jy/GLIP/OUTPUT2/0.07466/",#10 07too good
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.3984633510574706/",#416
               # "/home/data/jy/GLIP/COMP_OUTPUTS/0.4165950561751854/",  #425
               # "/home/data/jy/GLIP/OUTPUT1/0.47323/",
        "/home/data/jy/GLIP/OUTPUT2/0.34435/",
        "/home/data/jy/GLIP/OUTPUT2/0.35047/",
        "/home/data/jy/GLIP/OUTPUT2/0.47447/",


               # "OUTPUT1/GT",  # GT
               ]
    for image_id in os.listdir(submits[1]):
        allpic = ["000000000038.jpg","000000000226.jpg","000000000017.jpg","000000000256.jpg",
                  "000000000277.jpg","000000000454.jpg","000000000290.jpg","000000000206.jpg","000000000247.jpg","000000000365.jpg","000000000275.jpg",
                  "000000000289.jpg","000000000267.jpg","000000000453.jpg","000000000244.jpg","000000000167.jpg","000000000464.jpg",
                  "000000000458.jpg","000000000460.jpg",]
        allpic = ["000000000053.jpg","000000000023.jpg",
                  "000000000003.jpg","000000000022.jpg","000000000017.jpg","000000000010.jpg","000000000034.jpg","000000000021.jpg","000000000030.jpg",
                  "000000000014.jpg","000000000024.jpg"]
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
        TITLE = 'abcdefghijklmnopq'
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
            plt.savefig('TOSHOWtmi/ALLcom{}_consep.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWtmi/ALLcom_consep.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_ALLSHOW_COMPARE():
    fid=0
    submits = [
        "/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
               "/home/data/jy/GLIP/OUTPUTcoco_/0.50102/",
               "/home/data/jy/GLIP/OUTPUTcoco_/0.50290/",
                "/home/data/jy/GLIP/OUTPUTcoco_/0.50067/",
                "/home/data/jy/GLIP/OUTPUTcoco_/0.46908/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.47028/",#5
               "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.34480/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.31844/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.29408/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.28594/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.21185/",#10
                "/home/data/jy/GLIP/OUTPUTcoco_/0.12638/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.14493/",
                "/home/data/jy/GLIP/OUTPUTcoco_/0.01097/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.07739/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12297/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12521/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12529/",
               ]
    for image_id in os.listdir(submits[1])[:700]:
        allpic = ["000000159977.jpg","000000000027.jpg",
                  "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
                  "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
                  "000000000456.jpg","000000000023.jpg",]
        allpic = ["000000000257.jpg","000000000463.jpg","000000000180.jpg","000000000023.jpg",]
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
        fig.set_size_inches(2*INCH, 1.4*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopq'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.02,hspace=0.02)
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
                plt.title('{}'.format(iid), y=0.1)
            except:
                print('miss image')
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.margins(0, 0)
        plt.title(image_id[:20] + '.jpg', y=0)
        if fid%H==0:
            plt.savefig('TOSHOWcvpr2025/ALLcom{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.savefig('TOSHOWtmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_ALLSHOW_COMPARE_FINALMAP():
    fid=0
    submits = [
        "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",#"/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50102/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50290/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50067/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.46908/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.47028/",  # 5
        "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.34480/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.31844/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.29408/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.28594/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.21185/",  # 10
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12638/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.14493/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.01097/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.07739/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12297/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12521/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12529/",
    ]
    for image_id in os.listdir(submits[1])[:700]:
        allpic = ['000000159977', '000000293390', '000000055150', '000000529122', '00000000336658',
                  '00000000014038', '000000388927', '000000442009', '000000126107', '000000112298',
                  '000000358525', '00000190648','000000289393','000000335081','000000166664',
                  '000000557172','000000302165','000000146457'
                  ]
        allpic_batch2 = [
            '000000029397', '000000021167', '000000015254', '000000009769',  # '000000025139',
        ]
        allpic_batch2 = [
            '000000020107', '000000013597', '000000023034', '000000009769',
        ]
        ranks = {allpic[0]: [0,14, 13,7, 7,7, 2], allpic[1]: [0, 16,12,14, 9,11,2], allpic[2]: [0, 15, 13, 11, 7, 18,16],allpic[3]: [0, 14,13,15, 10,11,2 ], allpic[4]: [0, 17, 18, 16, 3, 1,1],
                 allpic[5]: [0, 17, 19, 16, 14, 2,1], allpic[6]: [0, 18, 14,15,12, 17,2], allpic[7]: [0,14, 13,18, 9, 15, 5],allpic[8]: [0,14, 11,10,9,13,5], allpic[9]: [0, 17, 15, 11, 10,13,5],
                 allpic[10]: [0,17,16,15, 2,10,7], allpic[11]: [0,18,14,11, 15,2,1],allpic[12]: [0,12,14,5,3,13,2], allpic[13]: [0,12,14,17,3,16,2],allpic[14]: [0,12,16,18,3,2,11],
                 allpic[15]: [0,12,16,8,10,17,3],allpic[16]: [0,11,12,14,13,8,7],allpic[17]: [0,12,14,10,2,13,8],
                 }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 18
        if fid % H == 0:
            plt.close()
        COL_NUMS = 7
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(2 * INCH, 4 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefgh'
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/ALLcom.png')
    # map = io.imread('TOSHOWcvpr2025/ALLcom.png')
    # io.imsave('TOSHOWcvpr2025/ALLcom.png', map[:-200, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_ALLSHOW_COMPARE_rebuttal():
    fid=0
    submits = [
        "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",#"/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50102/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50290/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50067/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.46908/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.47028/",  # 5
        "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.34480/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.31844/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.29408/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.28594/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.21185/",  # 10
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12638/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.14493/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.01097/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.07739/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12297/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12521/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12529/",
    ]
    for image_id in os.listdir(submits[2])[:700]:
        allpic = ['000000434247', '000000297084', '000000512564', '000000528578', '00000000336658',
                  '000000014038', '000000388927', '000000442009', '000000126107', '000000112298',
                  '000000358525', '00000190648','000000289393','000000335081','000000166664',
                  '000000557172','000000302165','000000146457'
                  ]
        allpic_batch2 = [ '000000434247','000000512564',
                  ]
        ranks = {allpic[0]: [0,7,9, 12,13, 8], allpic[1]: [0, 16,12,14, 9,11,2], allpic[2]: [0, 13,14,12,7,9],allpic[3]: [0, 7,8,9,12,10 ], allpic[4]: [0, 17, 18, 16, 3, 1,1],
                 allpic[5]: [0, 17, 19, 16, 14, 2,1], allpic[6]: [0, 18, 14,15,12, 17,2], allpic[7]: [0,14, 13,18, 9, 15, 5],allpic[8]: [0,14, 11,10,9,13,5], allpic[9]: [0, 17, 15, 11, 10,13,5],
                 allpic[10]: [0,17,16,15, 2,10,7], allpic[11]: [0,18,14,11, 15,2,1],allpic[12]: [0,12,14,5,3,13,2], allpic[13]: [0,12,14,17,3,16,2],allpic[14]: [0,12,16,18,3,2,11],
                 allpic[15]: [0,12,16,8,10,17,3],allpic[16]: [0,11,12,14,13,8,7],allpic[17]: [0,12,14,10,2,13,8],
                 }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic_batch2:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 3
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.4 * INCH, 0.53 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefgh'
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                im_method = transform.resize(im_method, (480,640, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/rebuttal.png')
    map = io.imread('TOSHOWcvpr2025/rebuttal.png')
    io.imsave('TOSHOWcvpr2025/rebuttal.png', map[:-280, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_ALLSHOW_COMPARE_FINALMAP_afterselect():
    fid=0
    submits = [
        "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",#"/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50102/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50290/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.50067/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.46908/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.47028/",  # 5
        "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.34480/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.31844/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.29408/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.28594/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.21185/",  # 10
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12638/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.14493/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.01097/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.07739/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12297/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12521/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12529/",
    ]
    for image_id in os.listdir(submits[1])[:700]:
        allpic = ['000000159977', '000000293390', '000000055150', '000000529122', '00000000336658',
                  '00000000014038', '000000388927', '000000442009', '000000126107', '000000112298',
                  '000000358525', '00000190648','000000289393','000000335081','000000166664',
                  '000000557172','000000302165','000000146457'
                  ]
        allpic_batch2 = [ '000000293390', '000000055150', '000000289393','000000335081',
                  ]
        ranks = {allpic[0]: [0,14, 13,7, 7,7, 2], allpic[1]: [0, 16,12,14, 9,11,2], allpic[2]: [0, 15, 13, 11, 7, 18,16],allpic[3]: [0, 14,13,15, 10,11,2 ], allpic[4]: [0, 17, 18, 16, 3, 1,1],
                 allpic[5]: [0, 17, 19, 16, 14, 2,1], allpic[6]: [0, 18, 14,15,12, 17,2], allpic[7]: [0,14, 13,18, 9, 15, 5],allpic[8]: [0,14, 11,10,9,13,5], allpic[9]: [0, 17, 15, 11, 10,13,5],
                 allpic[10]: [0,17,16,15, 2,10,7], allpic[11]: [0,18,14,11, 15,2,1],allpic[12]: [0,12,14,5,3,13,2], allpic[13]: [0,12,14,17,3,16,2],allpic[14]: [0,12,16,18,3,2,11],
                 allpic[15]: [0,12,16,8,10,17,3],allpic[16]: [0,11,12,14,13,8,7],allpic[17]: [0,12,14,10,2,13,8],
                 }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic_batch2:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.4 * INCH, 0.88 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefgh'
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid+1]] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                im_method = transform.resize(im_method, (480,640, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/ALLcom.png')
    map = io.imread('TOSHOWcvpr2025/ALLcom.png')
    io.imsave('TOSHOWcvpr2025/ALLcom.png', map[:-280, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_ALLSHOW_COMPARE_FINALMAP0():
    fid=0
    submits = [
        "/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
               "/home/data/jy/GLIP/OUTPUTcoco_/0.50102/",
               "/home/data/jy/GLIP/OUTPUTcoco_/0.50290/",
                "/home/data/jy/GLIP/OUTPUTcoco_/0.50067/",
                "/home/data/jy/GLIP/OUTPUTcoco_/0.46908/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.47028/",#5
               "/home/data/jy/GLIP/OUTPUTcoco_/0.40764/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.40270/",
               "/home/data/jy/GLIP/OUTPUTcoco_/0.39611/",

               "/home/data/jy/GLIP/OUTPUTcoco_/0.35460/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.34432/",#10
               "/home/data/jy/GLIP/OUTPUTcoco_/0.20124/",
               "/home/data/jy/GLIP/OUTPUTcoco_/0.20406/",
               "/home/data/jy/GLIP/OUTPUTcoco_/0.10303/",
                "/home/data/jy/GLIP/OUTPUTcoco_/0.08326/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12297/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12521/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.12529/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.08694/",
        "/home/data/jy/GLIP/OUTPUTcoco_/0.05121/",
               ]
    for image_id in os.listdir(submits[1])[:200]:
        allpic = ['000000159977', '000000442746', '000000082696', '000000529122', '000000336658',
                  '000000014038', '000000388927', '000000442009', '000000126107', '000000112298',
                  '000000358525', '000000345397',
                  ]
        allpic_batch2 = [
            '000000029397', '000000021167', '000000015254', '000000009769',  # '000000025139',
        ]
        allpic_batch2 = [
            '000000020107', '000000013597', '000000023034', '000000009769',
        ]
        ranks = {allpic[0]: [0,10, 12,5, 1, 2], allpic[1]: [0, 16,18,13, 5,2], allpic[2]: [0, 17, 12, 5, 9, 13],allpic[3]: [0, 15,16, 12, 13,1 ], allpic[4]: [0, 17, 18, 16, 3, 1],
                 allpic[5]: [0, 17, 19, 16, 14, 2], allpic[6]: [0, 18,19, 10, 14, 1], allpic[7]: [8,16, 2,18, 17,  5],allpic[8]: [8,16, 19,17,12,11], allpic[9]: [0, 17, 18, 14, 13,5],
                 allpic[10]: [0,17,19,18, 13, 3], allpic[11]: [0,12,13,14,11, 1]
                 }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 12
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45 * INCH, 0.8 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid, submit in enumerate(submits[:6]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                # if fid<3 and iid==0:
                #     im_method=transform.resize(im_method,(1088,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/ALLcom.png')
    # map = io.imread('TOSHOWcvpr2025/ALLcom.png')
    # io.imsave('TOSHOWcvpr2025/ALLcom.png', map[:-200, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_CAM():
    fid=0
    submits = [
        "/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/TRAIN_0s/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",
        "/home/data/jy/GLIP/TRAIN_full/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter1000/",
               "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter2000/",
                "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter3000/",#5
                "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter4000/",
                "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter1000/",#
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter3000/",#10
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter3000/",#15
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot2/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0002000/cam_of_iter1/",#20
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0005000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0002000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter100/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter200/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter300/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter400/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter500/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",
               ]
    for image_id in os.listdir(submits[3])[:]:
        allpic = ["000000159977.jpg","000000000027.jpg",
                  "000000000046.jpg","000000000463.jpg","000000000174.jpg","000000000257.jpg","000000000363.jpg","000000000203.jpg","000000000234.jpg",
                  "000000000455.jpg","000000000352.jpg","000000000200.jpg","000000000036.jpg","000000000448.jpg","000000000180.jpg",
                  "000000000456.jpg","000000000023.jpg",]
        allpic = ["000000000257.jpg","000000000463.jpg","000000000180.jpg","000000000023.jpg",]
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
        fig.set_size_inches(4*INCH, 1.4*INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopq'
        for iid,submit in enumerate(submits[:]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid-1)%H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,wspace=0.02,hspace=0.02)
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
                plt.title('{}'.format(iid), y=0.1)
            except:
                print('miss image :{}'.format(submits[iid] + '/' + image_id))
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.margins(0, 0)
        plt.title(image_id[:20] + '.jpg', y=0)
        if fid%H==0:
            plt.savefig('TOSHOWcvpr2025/CAM/CAM{}.png'.format(fid))
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.savefig('TOSHOWtmi/ALLcom.png')
    # map=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/ALLcom.png',map[:-600,:,:])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list,key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list
def print_CVPR2025_CAMFINAL():
    fid=0
    submits = [
        "/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/TRAIN_0s/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",
        "/home/data/jy/GLIP/TRAIN_full/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter3000/",  # 5
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter1000/",  #
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter3000/",  # 10
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter3000/",  # 15
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot2/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0002000/cam_of_iter1/",  # 20
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0005000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0002000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter100/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter200/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter300/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter400/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter500/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",
    ]
    for image_id in os.listdir(submits[3])[:]:
        allpic = ['000000030785', '000000007108', '000000026564', '000000012670', '000000029640',
                  '000000026690', '000000000785', '000000005001', '000000025986',
                  ]
        allpic_batch2 = [
            '000000016228', '000000018576', '000000006954', '000000002587',  '000000017714','000000015079',
        ]
        allpic_batch2 = [
            '000000020107', '000000013597', '000000023034', '000000009769',
        ]
        ranks_batch2 = {allpic[0]: [0, 32,37, 5,2], allpic[1]: [0, 18, 50,58,2], allpic[2]: [0, 17,13,20,55],allpic[3]: [0,2 ,54,24,16], allpic[4]: [0, 61,18,42,4],
                        allpic[5]: [0, 57,56,61,5],
                 }
        ranks = {allpic[0]: [0,2,1,38, 56,10], allpic[1]: [0, 1,10,55,9,47], allpic[2]: [0, 2,13,57,43,40],allpic[3]: [0, 21,18,61,19,4 ], allpic[4]: [0, 1,16,61,9,42],
                 allpic[5]: [0, 2,5,3,4,8], allpic[6]: [0,1,4,5,2,3], allpic[7]: [0,43,1,50,2,17], allpic[8]: [0,2,1,58,5,17],
                 }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 10
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45 * INCH, 2 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                im_method=transform.resize(im_method,(800,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        plt.title(image_id[:20] + '.jpg', y=0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/FINAL_CAM.png')
    # map = io.imread('TOSHOWcvpr2025/ALLcom.png')
    # io.imsave('TOSHOWcvpr2025/ALLcom.png', map[:-200, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_CAMFINAL_iccvrebuttal():
    fid=0
    submits = [
        "/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/TRAIN_0s/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",
        "/home/data/jy/GLIP/TRAIN_full/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter3000/",  # 5
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter1000/",  #
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter3000/",  # 10
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter3000/",  # 15
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot2/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0002000/cam_of_iter1/",  # 20
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0005000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0002000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter100/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter200/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter300/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter400/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter500/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",
    ]
    for image_id in os.listdir(submits[3])[:]:
        allpic = ['000000030785', '000000007108', '000000026564', '000000012670', '000000029640',
                  '000000026690', '000000000785', '000000005001', '000000025986',
                  ]
        allpic_batch2 = [
            '000000016228', '000000018576', '000000006954', '000000002587',  '000000017714','000000015079',
        ]
        allpic_batch2 = [
            '000000025986', '000000005001', '000000030785', '000000007108',
        ]
        ranks_batch2 = {allpic[0]: [0, 32,37, 5,2], allpic[1]: [0, 18, 50,58,2], allpic[2]: [0, 17,13,20,55],allpic[3]: [0,2 ,54,24,16], allpic[4]: [0, 61,18,42,4],
                        allpic[5]: [0, 57,56,61,5],
                 }
        ranks = {allpic[0]: [0,2,1,38, 56,10], allpic[1]: [0, 1,10,55,9,47], allpic[2]: [0, 2,13,57,43,40],allpic[3]: [0, 21,18,61,19,4 ], allpic[4]: [0, 1,16,61,9,42],
                 allpic[5]: [0, 2,5,3,4,8], allpic[6]: [0,1,4,5,2,3], allpic[7]: [0,43,1,50,2,17], allpic[8]: [0,2,1,58,5,17],
                 }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic_batch2:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 4
        if fid % H == 0:
            plt.close()
        COL_NUMS = 3
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45 * INCH, 1 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                im_method=transform.resize(im_method,(800,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        plt.title(image_id[:20] + '.jpg', y=0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/FINAL_CAM_iccv.png')
    # map = io.imread('TOSHOWcvpr2025/ALLcom.png')
    # io.imsave('TOSHOWcvpr2025/ALLcom.png', map[:-200, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_CAMFINAL_RE():
    from skimage import io
    fid=0
    submits = [
        "/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/TRAIN_0s/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",
        "/home/data/jy/GLIP/TRAIN_full/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter3000/",  # 5
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter1000/",  #
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter3000/",  # 10
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter3000/",  # 15
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot2/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0002000/cam_of_iter1/",  # 20
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0005000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0002000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter100/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter200/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter300/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter400/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter500/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",
    ]
    for image_id in os.listdir(submits[3])[:]:
        allpic = ['000000030785', '000000007108', '000000026564', '000000012670', '000000029640',
                  '000000026690', '000000000785', '000000005001', '000000025986',
                  ]

        allpic_batch2 = ['000000000785','000000026564','000000012670','000000029640',
                  ]
        ranks_batch2 = {allpic[0]: [0, 32,37, 5,2], allpic[1]: [0, 18, 50,58,2], allpic[2]: [0, 17,13,20,55],allpic[3]: [0,2 ,54,24,16], allpic[4]: [0, 61,18,42,4],
                        allpic[5]: [0, 57,56,61,5],
                 }
        ranks = {allpic[0]: [0,2,1,38, 56,1,10], allpic[1]: [0, 1,10,55,9,1,47], allpic[2]: [0, 3,13,40,43,2,57],allpic[3]: [0, 21,18,61,19,3,4 ], allpic[4]: [0, 1,16,61,9,1,42],
                 allpic[5]: [0, 2,5,3,4,1,8], allpic[6]: [0,1,4,5,2,1,3], allpic[7]: [0,43,1,50,2,1,17], allpic[8]: [0,2,1,58,5,1,17],
                 }
        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic_batch2:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 5
        if fid % H == 0:
            plt.close()
        COL_NUMS = 7
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.41 * INCH, 1 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefgh'
        IMG_REFINE=[None,None,None,None,None,None,None,None,None]
        if '000000000785' in image_id:
            print(image_id)
            im2=io.imread(submits[2] + '/' + image_id)
            im2 = transform.resize(im2, (800, 800, 3))
            im3 = io.imread(submits[3] + '/' + image_id)
            im3 = transform.resize(im3, (800, 800, 3))
            im9 = io.imread(submits[9] + '/' + image_id)
            im9 = transform.resize(im9, (800, 800, 3))
            im11 = io.imread(submits[11] + '/' + image_id)
            im11 = transform.resize(im11, (800, 800, 3))
            im_combine=im2*0.9+im9*0.1
            im2[200:, :, :] = im_combine[200:, :, :]
            IMG_REFINE[5] = im2
            # plt.close()
            # plt.imshow(im_combine.astype(np.uint8))
            # plt.show()
        elif '000000005001' in image_id:
            print(image_id)
            im2=io.imread(submits[1] + '/' + image_id)
            im2 = transform.resize(im2, (800, 800, 3))
            im3 = io.imread(submits[3] + '/' + image_id)
            im3 = transform.resize(im3, (800, 800, 3))
            im9 = io.imread(submits[9] + '/' + image_id)
            im9 = transform.resize(im9, (800, 800, 3))
            im_combine=im2*0.4+im3*0.5+im9*0.1
            IMG_REFINE[5] = im_combine
            # plt.close()
            # plt.imshow(im_combine.astype(np.uint8))
            # plt.show()
        elif '000000029640' in image_id:
            print(image_id)
            im1=io.imread(submits[1] + '/' + image_id)
            im1 = transform.resize(im1, (800, 800, 3))
            im2 = io.imread(submits[2] + '/' + image_id)
            im2 = transform.resize(im2, (800, 800, 3))
            im9 = io.imread(submits[9] + '/' + image_id)
            im9 = transform.resize(im9, (800, 800, 3))
            im11 = io.imread(submits[11] + '/' + image_id)
            im11 = transform.resize(im11, (800, 800, 3))
            im16 = io.imread(submits[16] + '/' + image_id)
            im16 = transform.resize(im16, (800, 800, 3))
            im50 = io.imread(submits[50] + '/' + image_id)
            im50 = transform.resize(im50, (800, 800, 3))
            im_combine=im1*0.1+im2*0.4+im11*0.4+im16*0.1
            IMG_REFINE[5] = im16*0.5+im9*0.5
            IMG_REFINE[2] = im9*0.8+im50*0.2
            IMG_REFINE[3] = im11*0.5+im2*0.5
        elif '000000007108' in image_id:
            im1=io.imread(submits[1] + '/' + image_id)
            im1 = transform.resize(im1, (800, 800, 3))
            im2 = io.imread(submits[2] + '/' + image_id)
            im2 = transform.resize(im2, (800, 800, 3))
            im11 = io.imread(submits[11] + '/' + image_id)
            im11 = transform.resize(im11, (800, 800, 3))
            IMG_REFINE[5] = im1*0.2+im2*0.3+im11*0.5
            # IMG_REFINE[2] = im11 * 0.5 + im2 * 0.5
            # IMG_REFINE[3] = im1 * 0.5 + im2 * 0.5
        elif '00000026564' in image_id:
            im1=io.imread(submits[1] + '/' + image_id)
            im1 = transform.resize(im1, (800, 800, 3))
            im2 = io.imread(submits[2] + '/' + image_id)
            im2 = transform.resize(im2, (800, 800, 3))
            im11 = io.imread(submits[44] + '/' + image_id)
            im11 = transform.resize(im11, (800, 800, 3))
            im57 = io.imread(submits[57] + '/' + image_id)
            im57 = transform.resize(im57, (800, 800, 3))
            im57[600:,520:680,:]=im11[600:,520:680,:]
            im57[:200, 717:, :] = im1[:200, 717:, :]
            IMG_REFINE[6] = im57
            # IMG_REFINE[2] = im11 * 0.5 + im2 * 0.5
            # IMG_REFINE[3] = im1 * 0.5 + im2 * 0.5
        elif rank[5]==1:
            im1=io.imread(submits[rank[3]] + '/' + image_id)
            im1 = transform.resize(im1, (800, 800, 3))
            im2 = io.imread(submits[rank[4]] + '/' + image_id)
            im2 = transform.resize(im2, (800, 800, 3))
            im11 = io.imread(submits[rank[5]] + '/' + image_id)
            im11 = transform.resize(im11, (800, 800, 3))
            IMG_REFINE[5] = im1*0.2+im2*0.3+im11*0.5
            # IMG_REFINE[2] = im11 * 0.5 + im2 * 0.5
            # IMG_REFINE[3] = im1 * 0.5 + im2 * 0.5
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                im_method=transform.resize(im_method,(800,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                if IMG_REFINE[iid] is None:
                    plt.imshow(im_method[:, :, :])
                else:
                    plt.imshow(IMG_REFINE[iid])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # plt.title(image_id[:20] + '.jpg', y=0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('/home/data/jy/GLIP/TOSHOWcvpr2025/FINAL_CAM.png')
    map = io.imread('/home/data/jy/GLIP/TOSHOWcvpr2025/FINAL_CAM.png')
    io.imsave('/home/data/jy/GLIP/TOSHOWcvpr2025/FINAL_CAM.png', map[:-320, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_CAM148():
    fid=0
    submits = [
        "/home/data/jy/GLIP/ORI_WITH_BOX_ALLCLASS/",  # GT
        "/home/data/jy/GLIP/TRAIN_0s/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0030000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0035000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0040000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0045000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_fanew_IMPROMPT_model_0050000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_maple_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0005000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0010000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0015000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0020000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0025000_pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/PLOT__home_data_jy_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_upt_model_0030000_pth/",
        "/home/data/jy/GLIP/TRAIN_full/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter3000/",  # 5
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/glip_noval1s/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter1000/",  #
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter3000/",  # 10
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter2000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter3000/",  # 15
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter4000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext/cam_of_iter5000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours1shot2/cam_of_iter1000/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0002000/cam_of_iter1/",  # 20
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours2shot_0005000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0001000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0002000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0003000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/TRAIN_ours_notext3_0004000/cam_of_iter1/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter100/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter200/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter300/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter400/",
        "/home/data/jy/GLIP/PLOT2/glip_noval2s/cam_of_iter500/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data2_wyj_GLIP_OUTPUT_TRAIN_fanew_178to2447_model_0000400.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocoours110_model_0001200.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0010000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest_model_0005000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0090000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0080000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0070000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_cocobest3_model_0060000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0250000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0240000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0230000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0220000.pth/",
        "/home/data/jy/GLIP/PLOT2/OLD/_data1_wyj_GLIP_OUTPUT_TRAIN_OUTPUT_TRAIN_coco_ta_model_0200000.pth/",
    ]
    for image_id in os.listdir(submits[3])[:]:

        allpic = [
            '000000016228', '000000018576', '000000006954', '000000002587',  '000000017714','000000015079',
        ]
        allpic_batch2 = [
            '000000020107', '000000013597', '000000023034', '000000009769',
        ]
        ranks = {allpic[0]: [0, 32,37, 5,2], allpic[1]: [0, 18, 50,58,2], allpic[2]: [0, 17,13,20,55],allpic[3]: [0,2 ,54,24,16], allpic[4]: [0, 61,18,42,4],
                        allpic[5]: [0, 57,56,61,5],
                 }

        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = False
        for picid in allpic:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            if picid in image_id:
                print(picid)
                NEEDTOSHOW = True
                SHORTID = picid
                rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 10
        if fid % H == 0:
            plt.close()
        COL_NUMS = 5
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45 * INCH, 2 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefg'
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                im_method=transform.resize(im_method,(800,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(submits[rank[iid]] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        plt.title(image_id[:20] + '.jpg', y=0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/148_CAM.png')
    # map = io.imread('TOSHOWcvpr2025/ALLcom.png')
    # io.imsave('TOSHOWcvpr2025/ALLcom.png', map[:-200, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_ODINW13():
    fid=0
    submits = [#odinw1
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.22483/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.27664/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.27518/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.25135/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.23661/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.32425/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.39475/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.49883/",
    ]
    submits = [#odinw2
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.04482/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.04556/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.05711/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.05904/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06040/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06435/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06755/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06784/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.07671/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.07714/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.07912/",
    ]
    od3='/home/data/jy/GLIP/OUTPUTodinw3_val/'
    submits=os.listdir(od3)
    od4='/home/data/jy/GLIP/OUTPUTodinw4_val/'
    submits=os.listdir(od4)
    od5='/home/data/jy/GLIP/OUTPUTodinw5_val/'
    submits=os.listdir(od5)
    od6='/home/data/jy/GLIP/OUTPUTodinw6_val/'
    submits=os.listdir(od6)
    od7='/home/data/jy/GLIP/OUTPUTodinw7_val/'
    submits=os.listdir(od7)
    od8='/home/data/jy/GLIP/OUTPUTodinw8_val/'
    submits=os.listdir(od8)
    od9='/home/data/jy/GLIP/OUTPUTodinw9_val/'
    submits=os.listdir(od9)
    od10='/home/data/jy/GLIP/OUTPUTodinw10_val/'
    submits=os.listdir(od10)
    od11 = '/home/data/jy/GLIP/OUTPUTodinw11_val/'
    submits = os.listdir(od11)
    od12 = '/home/data/jy/GLIP/OUTPUTodinw12_val/'
    submits = os.listdir(od12)
    od13 = '/home/data/jy/GLIP/OUTPUTodinw13_val/'
    submits = os.listdir(od13)
    for id,submit in enumerate(submits):
        submits[id]=od13+submits[id]
    for image_id in os.listdir(submits[-1])[:20]:

        allpic = [
            "000000000125.jpg","000000000005.jpg", '000000000002.jpg', '000000000022.jpg', '000000000003.jpg',
            '000000002128.jpg','000000000000.jpg', '000000000694.jpg', '000000000028.jpg', '000000000007.jpg',
            '000000000010.jpg','000000000015.jpg', '000000000083.jpg',
        ]
        allpic_batch2 = [
            '000000020107', '000000013597', '000000023034', '000000009769',
        ]
        ranks = {allpic[0]: [0, 1,2,3,7,6], allpic[1]: [0,1,2,3,4,6,9], allpic[2]: [0, 1,2,3,4,13,16],allpic[3]: [0,1,2,3,4,5,6], allpic[4]: [0, 2,10,3,11,9,],
                        allpic[5]: [6,0,19,1,2,20,],allpic[6]: [0,1,2,3,4,5,6],allpic[7]: [1,0,12,14,2,13],allpic[8]: [9,12,14,18,19,20],allpic[9]: [11,12,13,16,20,19],
                 allpic[10]: [16,20 ,14,15, 19,21],allpic[11]: [0,2,4,16,18,17],allpic[12]: [0,16,1,2,3,19,20],
                 }

        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = True
        # for picid in allpic:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
        #     if picid in image_id:
        #         print(picid)
        #         NEEDTOSHOW = True
        #         SHORTID = picid
        #         rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
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
        fig.set_size_inches(1.45 * INCH, 2 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890912345667890'
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(submits[iid] + '/' + image_id)[:, :, :]
                # im_method=transform.resize(im_method,(800,800,3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
                plt.title(str(iid), y=0)
            except:
                print(submits[iid] + '/' + image_id)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.2, color='k', fontsize=40)
                print(TITLE[iid])
        if fid == H - 1:
            break
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    plt.savefig('TOSHOWcvpr2025/odinw.png')
    # plt.show()
    # map = io.imread('TOSHOWcvpr2025/ALLcom.png')
    # io.imsave('TOSHOWcvpr2025/ALLcom.png', map[:-200, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_ODINW13_final():
    fid=0
    ALL_SUBMITS=[]
    submits = [#odinw1
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.22483/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.27664/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.27518/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.25135/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.23661/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.32425/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.39475/",
        "/home/data/jy/GLIP/OUTPUTodinw1_val/0.49883/",
    ]
    ALL_SUBMITS.append(submits)
    submits = [#odinw2
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.04482/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.04556/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.05711/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.05904/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06040/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06435/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06755/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.06784/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.07671/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.07714/",
        "/home/data/jy/GLIP/OUTPUTodinw2_val/0.07912/",
    ]
    ALL_SUBMITS.append(submits)
    od3='/home/data/jy/GLIP/OUTPUTodinw3_val/'
    submits=os.listdir(od3)
    for id,submit in enumerate(submits):
        submits[id]=od3+submits[id]
    ALL_SUBMITS.append(submits)
    od4='/home/data/jy/GLIP/OUTPUTodinw4_val/'
    submits=os.listdir(od4)
    for id,submit in enumerate(submits):
        submits[id]=od4+submits[id]
    ALL_SUBMITS.append(submits)
    od5='/home/data/jy/GLIP/OUTPUTodinw5_val/'
    submits=os.listdir(od5)
    for id,submit in enumerate(submits):
        submits[id]=od5+submits[id]
    ALL_SUBMITS.append(submits)
    od6='/home/data/jy/GLIP/OUTPUTodinw6_val/'
    submits=os.listdir(od6)
    for id,submit in enumerate(submits):
        submits[id]=od6+submits[id]
    ALL_SUBMITS.append(submits)
    od7='/home/data/jy/GLIP/OUTPUTodinw7_val/'
    submits=os.listdir(od7)
    for id,submit in enumerate(submits):
        submits[id]=od7+submits[id]
    ALL_SUBMITS.append(submits)
    od8='/home/data/jy/GLIP/OUTPUTodinw8_val/'
    submits=os.listdir(od8)
    for id,submit in enumerate(submits):
        submits[id]=od8+submits[id]
    ALL_SUBMITS.append(submits)
    od9='/home/data/jy/GLIP/OUTPUTodinw9_val/'
    submits=os.listdir(od9)
    for id,submit in enumerate(submits):
        submits[id]=od9+submits[id]
    ALL_SUBMITS.append(submits)
    od10='/home/data/jy/GLIP/OUTPUTodinw10_val/'
    submits=os.listdir(od10)
    for id,submit in enumerate(submits):
        submits[id]=od10+submits[id]
    ALL_SUBMITS.append(submits)
    od11 = '/home/data/jy/GLIP/OUTPUTodinw11_val/'
    submits = os.listdir(od11)
    for id,submit in enumerate(submits):
        submits[id]=od11+submits[id]
    ALL_SUBMITS.append(submits)
    od12 = '/home/data/jy/GLIP/OUTPUTodinw12_val/'
    submits = os.listdir(od12)
    for id,submit in enumerate(submits):
        submits[id]=od12+submits[id]
    ALL_SUBMITS.append(submits)
    od13 = '/home/data/jy/GLIP/OUTPUTodinw13_val/'
    submits = os.listdir(od13)
    for id,submit in enumerate(submits):
        submits[id]=od13+submits[id]
    ALL_SUBMITS.append(submits)
    ALL_HEIGHTS=np.ones((14,))
    for line_id in range(13):
        allpic = [
            "000000000125.jpg","000000000005.jpg", '000000000002.jpg', '000000000022.jpg', '000000000003.jpg',
            '000000002128.jpg','000000000000.jpg', '000000000694.jpg', '000000000028.jpg', '000000000007.jpg',
            '000000000010.jpg','000000000015.jpg', '000000000083.jpg',
        ]
        image_id = allpic[line_id]
        ranks = {allpic[0]: [0, 1,2,3,7,6], allpic[1]: [0,1,2,3,4,6,9], allpic[2]: [0 ,2,3,4,13,6],allpic[3]: [0,1,2,3,4,5,6], allpic[4]: [0, 2,10,3,11,9,],
                        allpic[5]: [6,0,19,1,2,20,],allpic[6]: [0,1,2,3,4,5,6],allpic[7]: [1,0,12,14,2,13],allpic[8]: [9,12,14,18,19,20],allpic[9]: [11,12,13,16,20,19],
                 allpic[10]: [16,20 ,14,15, 19,21],allpic[11]: [0,2,4,16,18,17],allpic[12]: [0,16,1,2,3,19,20],
                 }

        # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
        #       8496:[0,1,2,5,3,8]}
        # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        # allpic = [204,]
        SHORTID = 0

        NEEDTOSHOW = True
        # for picid in allpic:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
        #     if picid in image_id:
        #         print(picid)
        #         NEEDTOSHOW = True
        #         SHORTID = picid
        #         rank = ranks[SHORTID]
        if not NEEDTOSHOW:
            continue
        #         #print(dices)
        # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
        #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        H = 14
        if fid % H == 0:
            plt.close()
        COL_NUMS = 6
        INCH = 20
        fig = plt.gcf()
        fig.set_size_inches(1.45 * INCH, 2 * INCH)
        HAS_ = 0

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 0
        TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890912345667890'
        submits=ALL_SUBMITS[line_id]
        rank=ranks[image_id]
        for iid, submit in enumerate(submits[:COL_NUMS]):
            idx += 1
            plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())


            im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
            h_,w_,c=im_method.shape
            im_method=transform.resize(im_method,(int(h_*800/w_),800,c))
            ALL_HEIGHTS[fid-1]=h_/w_
            # elif fid==3 :#and iid==0:
            #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
            plt.imshow(im_method[:, :, :])
                # plt.title(str(iid), y=0)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[iid]), y=-0.21, color='k', fontsize=40)
                print(TITLE[iid])
        # if fid == H - 1:
        #     break
        # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.title(image_id[:20]+'.jpg',y=0)
        plt.margins(0, 0)
        # if fid%H==0:
        #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    GRIDSPEC=True
    if GRIDSPEC:
        fid=0;plt.close()
        import matplotlib.gridspec as GS
        gs=GS.GridSpec(14,6,height_ratios=ALL_HEIGHTS)
        for line_id in range(13):
            allpic = [
                "000000000125.jpg","000000000005.jpg", '000000000002.jpg', '000000000022.jpg', '000000000003.jpg',
                '000000002128.jpg','000000000000.jpg', '000000000694.jpg', '000000000028.jpg', '000000000007.jpg',
                '000000000010.jpg','000000000015.jpg', '000000000083.jpg',
            ]
            image_id = allpic[line_id]
            ranks = {allpic[0]: [0, 1,2,3,7,6], allpic[1]: [0,1,2,3,4,6,9], allpic[2]: [0 ,2,3,4,13,6],allpic[3]: [0,1,2,3,4,5,6], allpic[4]: [0, 2,10,3,11,9,],
                            allpic[5]: [6,0,19,1,2,20,],allpic[6]: [0,1,2,3,4,5,6],allpic[7]: [1,0,12,14,2,13],allpic[8]: [9,12,14,18,19,20],allpic[9]: [11,12,13,16,20,19],
                     allpic[10]: [16,20 ,14,15, 19,21],allpic[11]: [0,2,4,16,18,17],allpic[12]: [0,16,1,2,3,19,20],
                     }

            # ranks={3183:[0,1,5,7,2,4],2535:[0,4,1,3,2,5],
            #       8496:[0,1,2,5,3,8]}
            # allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
            # allpic = [204,]
            SHORTID = 0

            NEEDTOSHOW = True
            # for picid in allpic:#!!!wyj:batch2 should be the finalmap after SELECTION DECISION of zhouyang and xuyan
            #     if picid in image_id:
            #         print(picid)
            #         NEEDTOSHOW = True
            #         SHORTID = picid
            #         rank = ranks[SHORTID]
            if not NEEDTOSHOW:
                continue
            #         #print(dices)
            # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
            #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
            #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
            H = 14
            if fid % H == 0:
                plt.close()
            COL_NUMS = 6
            INCH = 20
            fig = plt.gcf()
            fig.set_size_inches(0.91 * INCH, 2 * INCH)
            HAS_ = 0

            fid += 1
            # if fid > 13:
            #     fid = 13
            idx = 0
            TITLE = 'abcdefghijklmnopqrstuvwxyz1234567890912345667890'
            submits=ALL_SUBMITS[line_id]
            rank=ranks[image_id]
            for iid, submit in enumerate(submits[:COL_NUMS]):
                idx += 1
                # plt.subplot(H, COL_NUMS, idx + ((fid - 1) % H) * COL_NUMS)
                # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())

                im_method = io.imread(submits[rank[iid]] + '/' + image_id)[:, :, :]
                h_,w_,c=im_method.shape
                im_method=transform.resize(im_method,(int(h_*800/w_),800,c))
                ALL_HEIGHTS[fid-1]=h_/w_
                ax1 = fig.add_subplot(gs[idx-1 + ((fid - 1) % H) * COL_NUMS])
                fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
                # fig.tight_layout()
                # ax1.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
                ax1.axis('off')
                ax1.margins(0, 0)
                ax1.imshow(im_method)
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
                    # plt.title(str(iid), y=0)
                if fid == H - 1:
                    plt.title('({})'.format(TITLE[iid]), y=-0.21, color='k', fontsize=40)
                    print(TITLE[iid])
            # if fid == H - 1:
            #     break
            # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.title(image_id[:20]+'.jpg',y=0)
            plt.margins(0, 0)
            # if fid%H==0:
            #     plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOWcvpr2025/odinw.png')
    # plt.show()
    map = io.imread('TOSHOWcvpr2025/odinw.png')
    io.imsave('TOSHOWcvpr2025/odinw.png', map[:-250, :, :])
    # plt.show()
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_CVPR2025_VOCSPLITX3():
    SPLIT_DIRS=[
        ["/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT0/0.54849/000000000017.jpg","/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT0/0.54849/000000000022.jpg",
         "/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT1/0.45188/000000000252.jpg",
         ],
        [#"/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT1/0.45188/000000000143.jpg",
         "/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT1/0.45188/000000000158.jpg",
         "/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT1/0.45188/000000000234.jpg","/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT1/0.45188/000000000238.jpg",
        ],
        ["/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT2/0.39165/000000000560.jpg",
         "/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT2/0.39165/000000000601.jpg",
         "/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT2/0.39165/000000000624.jpg",#"/home/data/jy/GLIP/OUTPUTodinw6_val_SPLIT2/0.39165/000000000682.jpg"

        ],
    ]
    fid = 0
    H = 4
    if fid % H == 0:
        plt.close()
    COL_NUMS = 3
    INCH = 20
    fig = plt.gcf()
    fig.set_size_inches(0.8 * INCH, 0.7* INCH)
    # if fid > 13:
    #     fid = 13
    idx = 0
    TITLE = 'abcdefg'
    for line_id, SPLIT_ in enumerate(SPLIT_DIRS):
        fid += 1
        for col_id,IMPATH in enumerate(SPLIT_):
            plt.subplot(H, COL_NUMS, col_id+1 + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(IMPATH)[:, :, :]
                im_method = transform.resize(im_method, (480, 640, 3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(IMPATH)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[col_id]), y=-0.2, color='k', fontsize=40)
                print(TITLE[col_id])
    # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.title(image_id[:20] + '.jpg', y=0)
    plt.margins(0, 0)
    # plt.title(image_id[:20] + '.jpg', y=0)
    plt.savefig('TOSHOWcvpr2025/VOC_3SPLIT.png')
def print_CVPR2025_MEDICALX3():
    SPLIT_DIRS=[
        ["/home/data/jy/GLIP/OUTPUTlidc_2017_val/0.07585/000000000001.jpg","/home/data/jy/GLIP/OUTPUTdeep_2017_val/0.00000/000000000040.jpg",
         # "/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.31965/000000000038.jpg",
         '/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.24739/000000000290.jpg',
         ],
        ["/home/data/jy/GLIP/OUTPUTlidc_2017_val/0.28083/000000000001.jpg",
            "/home/data/jy/GLIP/OUTPUTdeep_2017_val/0.01024/000000000040.jpg",'/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.41554/000000000290.jpg',
         # "/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.33343/000000000038.jpg",
        ],
    ]
    fid = 0
    H = 3
    if fid % H == 0:
        plt.close()
    COL_NUMS = 3
    INCH = 20
    fig = plt.gcf()
    fig.set_size_inches(0.5 * INCH, 0.5* INCH)
    # if fid > 13:
    #     fid = 13
    idx = 0
    TITLE = 'abcdefg'
    for line_id, SPLIT_ in enumerate(SPLIT_DIRS):
        fid += 1
        for col_id,IMPATH in enumerate(SPLIT_):
            plt.subplot(H, COL_NUMS, col_id+1 + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(IMPATH)[:, :, :]
                im_method = transform.resize(im_method, (800,800, 3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
            except:
                print(IMPATH)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[col_id]), y=-0.17, color='k', fontsize=30)
                print(TITLE[col_id])
    # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.title(image_id[:20] + '.jpg', y=0)
    plt.margins(0, 0)
    # plt.title(image_id[:20] + '.jpg', y=0)
    plt.savefig('TOSHOWcvpr2025/MEDICALX3.png')
    map = io.imread('TOSHOWcvpr2025/MEDICALX3.png')
    io.imsave('TOSHOWcvpr2025/MEDICALX3.png', map[:-280, :, :])
def print_CVPR2025_MONU():
    SPLIT_DIRS=['/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.24739/','/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.41554/',
        # ["/home/data/jy/GLIP/OUTPUTlidc_2017_val/0.25755/000000000001.jpg","/home/data/jy/GLIP/OUTPUTdeep_2017_val/0.00189/000000000002.jpg",
        #  # "/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.31965/000000000038.jpg",
        #  ],
        # ["/home/data/jy/GLIP/OUTPUTlidc_2017_val/0.28083/000000000120.jpg",
        #     "/home/data/jy/GLIP/OUTPUTdeep_2017_val/0.01024/000000000040.jpg",
        #  # "/home/data/jy/GLIP/OUTPUTcoco1_2017_val/0.33343/000000000038.jpg",
        # ],
    ]
    IM_IDS=[16,18,20,22,43,44,45,160,161,270,271,272,290,291,292,293,360,362,362,460,462,462]
    IM_IDS = [44, 271, 272, 290]
    fid = 0
    H = 3
    if fid % H == 0:
        plt.close()
    COL_NUMS = 4
    INCH = 20
    fig = plt.gcf()
    fig.set_size_inches(1 * INCH, 0.75* INCH)
    # if fid > 13:
    #     fid = 13
    idx = 0
    TITLE = 'abcdefghijklmnopqrstuvwxyz12345678090'
    for line_id, SPLIT_ in enumerate(SPLIT_DIRS):
        fid += 1
        for col_id,IMPAT in enumerate(IM_IDS):
            IMPATH=SPLIT_+"%012d.jpg" % (IMPAT)
            plt.subplot(H, COL_NUMS, col_id+1 + ((fid - 1) % H) * COL_NUMS)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.02)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            try:
                im_method = io.imread(IMPATH)[:, :, :]
                im_method = transform.resize(im_method, (800,800, 3))
                # elif fid==3 :#and iid==0:
                #     im_method = transform.resize(im_method,(1088,800,3))#transform.resize(im_method, (1344, 768, 3))
                plt.imshow(im_method[:, :, :])
                # plt.title('({})'.format(IMPAT), y=-0.15, color='k', fontsize=40)
            except:
                print(IMPATH)
            if fid == H - 1:
                plt.title('({})'.format(TITLE[col_id]), y=-0.15, color='k', fontsize=40)
                print(TITLE[col_id])
    # plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.title(image_id[:20] + '.jpg', y=0)
    plt.margins(0, 0)
    # plt.title(image_id[:20] + '.jpg', y=0)
    plt.savefig('TOSHOWcvpr2025/MONU.png')
    map = io.imread('TOSHOWcvpr2025/MONU.png')
    io.imsave('TOSHOWcvpr2025/MONU.png', map[:-420, :, :])
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
print_CVPR2025_CAMFINAL_iccvrebuttal()