import os
import numpy as np
# dip=os.listdir('/data1/wyj/M/samples/lung/TOSHOW3_BIGDATA2/') 0.75
dip=os.listdir('/data1/wyj/M/samples/lung/TOSHOW3_BIGDATA2/')
arraya=[]
for fname in dip:
    dice=float(fname[fname.find('_acc')+4:-4])
    if dice>0.85:
        arraya.append(dice)
        print('filename:{}  acc:{} '.format(fname[:23],dice))
        # output=os.popen("find /data3/datasets/KIDNEY_NP_BIGDATA -name '*{}*'".format(fname[:23])).read()
        # if output != '':
        #     # print("exists!!!!")
        #     print('filename:{}'.format(fname[:23]))
        #     # print(output)
print('average acc:{}'.format(np.mean(np.array(arraya))))