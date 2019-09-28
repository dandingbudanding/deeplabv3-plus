# coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from osgeo import gdal
import numpy as np
import math
np.set_printoptions(threshold=np.inf)#使print大量数据不用符号...代替而显示所有

img_w = 512
img_h = 512

image_sets = ['image_10.png','image_11.png','image_20.png','image_21.png']
image_sets_label = ['image_10_label.png','image_11_label.png','image_20_label.png','image_21_label.png']
path1 = os.path.abspath('..')  # 获取上一级目录
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb,yb,angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb

def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(30): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb,yb):
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)

    # if np.random.random() < 0.25:
    #     xb = blur(xb)
    #
    # if np.random.random() < 0.2:
    #     xb = add_noise(xb)

    return xb,yb

def gdal_convert_rgb(filepath = ''):
    print('convert to gdal mode...')
    dataset = gdal.Open(filepath)
    print(filepath)
    cols = dataset.RasterXSize  # 图像长度
    rows = (dataset.RasterYSize)  # 图像宽度
    print('cols = ', cols, 'rows = ', rows)

    xoffset = cols / 2
    yoffset = rows / 2

    col = math.floor(xoffset / 2)
    row = math.floor(yoffset / 2)

    band = dataset.GetRasterBand(3)  # 取第三波段

    r = band.ReadAsArray(int(xoffset), int(yoffset), int(col), int(row))  # 从数据的中心位置位置开始，取row行col列数据

    band = dataset.GetRasterBand(2)
    g = band.ReadAsArray(int(xoffset), int(yoffset), int(col), int(row))

    band = dataset.GetRasterBand(1)
    b = band.ReadAsArray(int(xoffset), int(yoffset), int(col), int(row))

    img2 = cv2.merge([r, g, b])
    return img2


def gdal_convert_single(filepath=''):
    print('convert to gdal mode...')
    dataset = gdal.Open(filepath)
    print(filepath)
    cols = dataset.RasterXSize  # 图像长度
    rows = (dataset.RasterYSize)  # 图像宽度
    print('cols = ', cols, 'rows = ', rows)

    xoffset = cols / 2
    yoffset = rows / 2

    col = math.floor(xoffset / 2)
    row = math.floor(yoffset / 2)

    band = dataset.GetRasterBand(1)
    b = band.ReadAsArray(int(xoffset), int(yoffset), int(col), int(row))

    img2 = b
    return img2

def creat_dataset(image_num = 20000, mode = 'augment'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    #print((path1+'/data/jingwei_round1_train_20190619/src/%d.jpg' %g_count))
    for i in tqdm(range(len(image_sets))):
        count = 0

        src_img = gdal_convert_rgb(path1+'/data/train/jingwei_round2_train_20190726/' + image_sets[i])  # 3 channels
        label_img = gdal_convert_single(path1+'/data/train/jingwei_round2_train_20190726/' + image_sets_label[i])  # single channel
        X_height,X_width,_ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w,:]
            label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi,label_roi = data_augment(src_roi,label_roi)

            # visualize = np.zeros((400,400)).astype(np.uint8)
            # visualize = label_roi *50
            #
            # cv2.imwrite(('F:/tianchi challenge/dataset/unet_train/visualize/%d.png' % g_count),visualize)
            print ("write to %s" %(path1+'/data/train/src/%d.jpg' %g_count))
            cv2.imwrite((path1+'/data/train/src/%d.jpg' %g_count),np.array(src_roi))
            print("write to %s" % (path1 + '/data/train/label/%d.png' % g_count))
            cv2.imwrite((path1+'/data/train/label/%d.png' %g_count),np.array(label_roi))
            count += 1
            g_count += 1



if __name__=='__main__':
    creat_dataset(mode='augment')



