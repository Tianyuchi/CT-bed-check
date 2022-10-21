import numpy as np
import os
import cv2
from PIL import Image
file_path=r'C:\Users\Z004F4FY\Desktop\New folder (3)'
out_path=r'C:\Users\Z004F4FY\Desktop\New folder (5)'
def Crop_Image(img_array):
    shape = img_array.shape
    #img_array = np.reshape(img_array, (shape[0], shape[1]))  # 获取array中的height和width
    high_window= img_array.shape[0]
    width_window = img_array.shape[1]
    cropped_img = img_array[0: width_window-100, 0: high_window, :]

    #cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return cropped_img

for file in os.listdir(file_path):
    file_dir=os.path.join(file_path,file)
    image_data= cv2.imread(file_dir)
    image_data = Crop_Image(image_data)
    out_filename = (out_path + '/' + file)
    print(out_filename)
    cv2.imwrite(out_filename,image_data)  # 第一个参数为保存的地址和文件名，第二个参数为保存的格式。