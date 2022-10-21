import os, sys
import random
import time

import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
from tkinter import ttk
import numpy as np
from dict_learn import DetectionAsDict
from utils_io import TopBed
import pydicom
import cv2
import os
import PIL
import PIL.Image

def load(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    return dcm

def defect_made(array):
    height, width = array.shape
    horizontal=random.randint(-150, 150)
    vertical=random.randint(0, 250)
    # 声明变换矩阵 参数1向右平移50个像素(左侧为负值)， 参数2向下平移100个像素（上移为负值）
    M = np.float32([[1, 0, horizontal], [0, 1, vertical]])
    # 进行2D 仿射变换
    shifted = cv2.warpAffine(array, M, (width, height))
    return shifted
def bbox2txt(mask_array,txt_name):
    y, x = np.nonzero(mask_array)
    #bbox = [x.min() - 1, y.min() - 1, x.max() + 1, y.max() + 1,1]
    f = open(os.path.join(txt_out,txt_name) +'.txt', mode='w')  # 打开文件，若文件不存在系统自动创建。
    # 参数name 文件名，mode 模式。 # w+ 可读可写  r+可读可写 a+可读可追加; w 只能操作写入  r 只能读取   a 向文件追加; # w模式打开文件，如果文件中有数据，再次写入内容，会把原来的覆盖掉;# wb+写入进制数据
    f.write(str(x.min() - 1)+' '+str(y.min() - 1)+' '+str(x.max() + 1)+' '+str(y.max() + 1)+' '+str(1))  # write 写入
    #f.writelines(x.min() - 1,, y.min() - 1, x.max() + 1, y.max() + 1,1)  # writelines()函数 会将列表中的字符串写入文件中，但不会自动换行，如果需要换行，手动添加换行符
    # 参数 必须是一个只存放字符串的列表
    f.close()  # 关闭文件

#template模板dicom和对应的mask
dcm_template=load(dcm_path=r'C:\Users\Z004F4FY\Desktop\bed-ot\1.3.12.2.1107.5.1.4.91603.30000021051406333654900000414\dicom')
mask_dir = r'C:\Users\Z004F4FY\Desktop\bed-ot\1.3.12.2.1107.5.1.4.91603.30000021051406333654900000414\error_image.png'
mask_template = cv2.imread(mask_dir,flags=-1)
####

###
file_path=r'D:\projects\Bed-workspace\negative_clean'#要进行瑕疵生成的图像根文件夹
output_path=r'C:\Users\Z004F4FY\Desktop\artificial-sample'#生成瑕疵数据输出的文件夹
txt_out=r'C:\Users\Z004F4FY\Desktop\artificial-txt'#瑕疵点的bounding box坐标信息txt文件的输出的文件夹
####

num=0
for subfile_name in os.listdir(file_path):
    subfile_path = os.path.join(file_path, subfile_name)
    for file_name in os.listdir(subfile_path):
        if file_name == 'dicom':
            num += 1
            print(num)
            file = os.path.join(subfile_path, file_name)
            # 要进行瑕疵生成的图像
            dcm_shift = load(dcm_path=file)
            dcm_shift0= dcm_shift.copy()
            for i in range(0, 1):
                #生成瑕疵图像
                shift_mask=defect_made(mask_template)#用于变换的mask
                dcm_shift0.pixel_array[np.nonzero(shift_mask)]=(dcm_template.pixel_array)[np.nonzero(mask_template)]
                dcm_shift0.PixelData=dcm_shift0.pixel_array.tobytes()
                new_name='artificial_defect_white_'+str(num)
                #dcm_shift0.save_as(os.path.join(output_path,new_name))   #存储为dcm格式
                #print(os.path.join(output_path,new_name))
                try:
                    WL_WW_map = {1: (-49, 40), 2: (-45, 40), 3: (-45, 40), 4: (-45, 40)}
                    WL, WW = WL_WW_map[int(dcm_shift0.SeriesNumber)]
                    array = dcm_shift0.pixel_array * float(dcm_shift0.RescaleSlope) + float(dcm_shift0.RescaleIntercept)
                    array.astype(np.float64)
                    array = (array - (WL - WW / 2)) / WW
                    array = np.clip(array * 255, 0, 255).astype(np.uint8)
                    img = PIL.Image.fromarray(array).convert('RGB')
                    img.save(os.path.join(output_path,new_name)+'.jpg')
                    bbox2txt(shift_mask,new_name)
                except KeyError:
                    print(file)