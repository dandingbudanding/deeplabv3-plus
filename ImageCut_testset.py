# coding=utf-8
from osgeo import gdal
from gdalconst import *
import numpy as np
import gdal
from osgeo import gdal_array
import os
import threading
import glob
import cv2
import math

def readTifImage(img_path, tl_x, tl_y, br_x, br_y, label_flag, flag):
    data = []
    # 以只读方式打开遥感影像
    dataset = gdal.Open(img_path, GA_ReadOnly)
    if dataset is None:
        print("Unable to open image file.")
        return data
    else:
        print("Open image file success.")
        bands_num = dataset.RasterCount
        print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        print(bands_num.__str__() + " bands in total.")
        if flag == 0:
            row_position_start = tl_y
            col_position_start = tl_x
            row_position_end = br_y - tl_y
            col_position_end = br_x - tl_x
        else:
            row_position_start = 0
            col_position_start = 0
            band_0 = dataset.GetRasterBand(1)
            row_position_end = band_0.YSize
            col_position_end = band_0.XSize
            print("size %d %d" % (row_position_end, col_position_end))

        if label_flag == 0:
            band_i = dataset.GetRasterBand(3)
            x = band_i.XSize
            y = band_i.YSize
            r = band_i.ReadAsArray(col_position_start, row_position_start, col_position_end, row_position_end)
            # r = band_i.ReadAsArray(7000, 7000, 500,500)
            print("Image merge success-1")
            band_i = dataset.GetRasterBand(2)
            g = band_i.ReadAsArray(col_position_start, row_position_start, col_position_end, row_position_end)
            # g = band_i.ReadAsArray(7000, 7000, 500,500)
            print("Image merge success-2")
            band_i = dataset.GetRasterBand(1)
            b = band_i.ReadAsArray(col_position_start, row_position_start, col_position_end, row_position_end)
            # b = band_i.ReadAsArray(7000, 7000, 500,500)
            print("Image merge success-3")
            img = cv2.merge([r, g, b])
            size = img.shape
            print(size)
            print("Image merge success")
            print("row position %d" % row_position_end)
            return img
        else:
            band_i = dataset.GetRasterBand(1)
            r = band_i.ReadAsArray(col_position_start, row_position_start, col_position_end, row_position_end)
            img = cv2.merge([r, r, r])
            size = img.shape
            print(size)
            print("Image merge success")
            print("row position %d" % row_position_end)
            return img

def cut(savefilepath, src, colNum, rowNum, num, index):
    x = src.shape[1]
    y = src.shape[0]
    if src.shape[1] % colNum != 0:
        imageNum_col = int(src.shape[1] / colNum + 1)
    else:
        imageNum_col = int(src.shape[1] / colNum)

    if src.shape[0] % rowNum != 0:
        imageNum_row = int(src.shape[0] / rowNum + 1)
    else:
        imageNum_row = int(src.shape[0] / rowNum)
    for i in range(0, imageNum_row):
        for j in range(0, imageNum_col):
            if j == imageNum_col - 1 and i != imageNum_row - 1:
                r = np.zeros((rowNum, colNum))
                g = np.zeros((rowNum, colNum))
                b = np.zeros((rowNum, colNum))
                img = cv2.merge([r, g, b])
                #img = r
                sub_image_1 = src[i*rowNum:i*rowNum + rowNum, j*colNum:src.shape[1]]
                img[0:rowNum, 0:src.shape[1]-j*colNum] = sub_image_1
                num += 1
                cv2.imwrite(savefilepath + ("/%d.jpg" % (num)), img)
                print("save image success, %d.jpg" % num)
            elif i == imageNum_row - 1 and j != imageNum_col - 1:
                r = np.zeros((rowNum, colNum))
                g = np.zeros((rowNum, colNum))
                b = np.zeros((rowNum, colNum))
                img = cv2.merge([r, g, b])
                #img = r
                sub_image_2 = src[i*rowNum:src.shape[0], j*colNum:j*colNum + colNum]
                img[0:src.shape[0]-i*rowNum, 0:colNum] = sub_image_2
                num += 1
                cv2.imwrite(savefilepath + ("/%d.jpg" % (num)), img)
                print("save image success, %d.jpg" % num)
            elif i == imageNum_row - 1 and j == imageNum_col - 1:
                r = np.zeros((rowNum, colNum))
                g = np.zeros((rowNum, colNum))
                b = np.zeros((rowNum, colNum))
                img = cv2.merge([r, g, b])
                #img = r
                sub_image_3 = src[i*rowNum:src.shape[0], j*colNum:src.shape[1]]
                img[0:src.shape[0] - i * rowNum, 0:src.shape[1]-j*colNum] = sub_image_3
                num += 1
#                cv2.imwrite((savefilepath + "\\%d.jpg") % (index, num), img)
                cv2.imwrite(savefilepath + ("/%d.jpg" % (num)), img)
                print("save image success, %d.jpg" % num)
            else:
                r = np.zeros((rowNum, colNum))
                g = np.zeros((rowNum, colNum))
                b = np.zeros((rowNum, colNum))
                img = cv2.merge([r, g, b])
                #img = r
                sub_image = src[i*rowNum:i*rowNum + rowNum, j*colNum:j*colNum + colNum]
                img[0:rowNum, 0:colNum] = sub_image
                num += 1
                cv2.imwrite(savefilepath + ("/%d.jpg" % (num)), img)
                print("save image success, %d.jpg" % num)

def thread_cut(img_path, tl_x, tl_y, br_x, br_y, label_flag, flag,
               savefilepath, colNum, rowNum, num, index, batchsize):
    #用index区分不同线程
    print("%d, %d, %d, %d" % (tl_x, tl_y, br_x, br_y))
    src = readTifImage(img_path, tl_x, tl_y, br_x, br_y, label_flag, flag)

    cut(savefilepath, src, colNum, rowNum, num + batchsize*index, index)


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False

if __name__ == '__main__':

    threadNum = 32#开启线程数
    label_flag = 0  # 读取非label图
    flag = 0  # 允许设置读取区域
    block_XSize = 512
    block_YSize = 512
    FolderPath = os.path.abspath('..') + "/data/test/jingwei_round2_test_b_20190830/"
    path_file_number = glob.glob(FolderPath + "\\*.png")  # 指定文件下个数
    print(len(path_file_number))

    file_list = []
    F = os.listdir(FolderPath)
    for f in F:
        if os.path.splitext(f)[1] == ".png":
            file_list.append(f)

    for i in file_list:
        num = -1
        threads = []
        Saves = []
        filePath = FolderPath + i
        dataset = gdal.Open(filePath, GA_ReadOnly)
        index = 0  # 线程数字ID
        if dataset is None:
            print("Unable to open image file for size.")
        else:
            col = dataset.RasterXSize  # col对应x方向
            row = dataset.RasterYSize
            row_expand = row + (block_YSize - row % block_YSize) #扩展行数为block_row的倍数
            thread_blockNum = row_expand / block_YSize // threadNum  #thread_blockNum 先把能够整除threadNum数量的block行数分配给每个线程

            savefilepath = FolderPath + "tailor/%s" %(i[-5:-4])
            mkdir(savefilepath)
            # 每个线程自己单独拥有一个存储路径 包括处理未整除行数的线程 因此线程数+1
            #for childThrNum in range(0, threadNum + 1):
            #    savefilepath = "F:\\tianchi_challenge\\AI\\test_round2\\%d" % i
            #    savefilepath += "\\thread%d" % childThrNum
            #    mkdir(savefilepath)
            #    Saves.append(savefilepath)

            batchsize = int(thread_blockNum*(col//block_XSize + 1))
            for childThrNum in range(0, threadNum):
                t = threading.Thread(target=thread_cut, args=(
                    filePath, 0, int(childThrNum*thread_blockNum*block_YSize), col, int(min((childThrNum+1)*thread_blockNum*block_YSize, row)), label_flag, flag,
                    #Saves[childThrNum], block_XSize, block_YSize, num, index, batchsize))
                    savefilepath, block_XSize, block_YSize, num, index, batchsize))
                threads.append(t)
                index = index + 1

            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if (row - threadNum*thread_blockNum*block_YSize) > 0:
                thread_cut(filePath, 0, int(threadNum*thread_blockNum*block_YSize), col, row, label_flag, flag,
                        savefilepath, block_XSize, block_YSize, num, index, batchsize)


