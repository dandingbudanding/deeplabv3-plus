import numpy as np
import cv2
import os
import math

subimg_w = 512
subimg_h = 512

# img_w=37241    #1:37241      2:25936
# img_h=19903    #1:19903      2:28832


image_sets = ['0.tif','1.tif','2.tif','3.tif','4.tif','5.tif','6.tif','7.tif','8.tif','9.tif']
path="./dataset/test/test/"
pathtosave="./dataset/test/test/testsrc/"
path1 = os.path.abspath('..')  # 获取上一级目录

pathtopredict=path1+"/data/test/jingwei_round2_test_b_20190830/inferenceresult/"
pathtopredicttosave=path1+"/submit/"



# def seperate():
#     g_count = 0
#     for i in range(len(image_sets)):
#         img_ori=cv2.imread("./dataset/src/src"+image_sets[i],-1)
#         img_orilabel = cv2.imread("./dataset/label/lab/" + image_sets[i], -1)
#         #img_ori=img_ori[:][:][0:3]
#         for h in range(int(img_h/subimg_h)):
#             for w in range(int(img_w / subimg_w)):
#                 imgsub=img_ori[h*subimg_h:h*subimg_h+subimg_w,w*subimg_w:w*subimg_w+subimg_w,0:3]
#                 imgsub_lab = img_orilabel[h*subimg_h:h*subimg_h+subimg_w,w*subimg_w:w*subimg_w+subimg_w]
#                 cv2.imwrite(("./dataset/src/src1/"+'%d.jpg' %g_count),imgsub)
#                 cv2.imwrite(("./dataset/label/label1/" + '%d.png' % g_count), imgsub_lab)
#                 g_count=g_count+1
#
# def seperatetest():
#     g_count = 0
#     for i in range(len(image_sets)):
#         img_ori=cv2.imread(path+image_sets[i],-1)
#         #img_ori=img_ori[:][:][0:3]
#         for h in range(int(img_h/subimg_h)):
#             for w in range(int(img_w / subimg_w)):
#                 imgsub=img_ori[h*subimg_h:h*subimg_h+subimg_w,w*subimg_w:w*subimg_w+subimg_w,0:3]
#                 cv2.imwrite((pathtosave+'%d.jpg' %g_count),imgsub)
#                 g_count=g_count+1

def combine(imagenum='3'):
    if imagenum == '3':
        img_w = 42614  # 3:37241      4:25936
        img_h = 20767  # 3:19903      4:28832
    elif imagenum == '4':
        img_w = 35055  # 3:37241      4:25936
        img_h = 29003  # 3:19903      4:28832
    elif imagenum=='5':
        img_w=43073
        img_h=20115
    elif imagenum=='6':
        img_w=62806
        img_h=21247
    combinenum = np.int(math.ceil(img_w / subimg_w) * math.ceil(img_h / subimg_h))
    imagename,imgnum=[],[]
    for f in os.listdir(pathtopredict+str(imagenum)):
        imagename.append(f)
        imgnum.append(f[:-9])
    img_ww=math.ceil(img_w/subimg_w)*subimg_w
    img_hh=math.ceil(img_h/subimg_h)*subimg_h
    combineimg = np.zeros((img_hh, img_ww)).astype(np.uint8)
    a,b=combineimg.shape
    combineimg_result = np.zeros((img_h, img_w)).astype(np.uint8)
    g_count=0

    for i in range(len(imagename)):
        img_ori=cv2.imread(pathtopredict+str(imagenum)+'/'+str(g_count)+".png",0)
        img_ori=img_ori.reshape((512,512))
        print(img_ori.shape)
        X_height,X_width=img_ori.shape
        a = img_ori[:, :]

        if i%(combinenum-1)!=0 or i==0:
            left = int(math.floor(i % math.ceil(img_w / subimg_w)) * subimg_w)
            up = int(math.floor(i / math.ceil(img_w / subimg_w)) * subimg_h)
            combineimg[up:up + subimg_h, left:left + subimg_w] = a
            g_count=g_count+1
            print(g_count)
        else:
            combineimg_result = combineimg[0:img_h, 0:img_w]
            cv2.imwrite((pathtopredicttosave +'image_'+str(imagenum)+ '_predict.png' ), combineimg_result)
            #cv2.imshow(combineimg_result)

if __name__=='__main__':
    combine('5')
