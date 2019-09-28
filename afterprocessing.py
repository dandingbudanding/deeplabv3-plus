# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:01:43 2019

@author: Megvii
"""

import cv2
import numpy as np
from skimage import morphology
import os

path1 = os.path.abspath('..')  # 获取上一级目录

def afterprocessing(imagenum='3'):
    print("start "+str(imagenum)+" afterprocessing")
    img_ori=cv2.imread(path1+"/data/test/jingwei_round2_test_b_20190830/image_"+str(imagenum)+".png",0)
    ret,img_ori=cv2.threshold(img_ori,0,255,cv2.THRESH_BINARY)#

    img_mask=cv2.imread(path1+"/submit/image_"+str(imagenum)+"_predict.png",0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))  # 矩形结构
    closing = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)  # 闭运算

    height,width=img_ori.shape
    closing=np.uint8(np.array(closing)/50)
    print("end1")
    img_result=closing & img_ori
    cv2.imwrite(path1+"/submit/image_"+str(imagenum)+"_predict.png",img_result)
    print("end2")

if __name__=='__main__':
    afterprocessing(imagenum='6')
