import os, sys
import argparse
import time
import numpy as np
import scipy
import scipy.ndimage
import scipy.ndimage.filters
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import imageio
import sklearn
import sklearn.decomposition 
import PIL
import PIL.Image
import PIL.ImageDraw
from utils_io import TopBed
import cv2
# def sobel(img):
#     return (scipy.ndimage.sobel(img, 0)**2 \
#             + scipy.ndimage.sobel(img, 1)**2)**0.5

class DetectionAsDict(object):
    def __init__(self, load=None):
        # self.recon_edge = False
        self.patch_size = (8,8)

        #self.dict_learner = sklearn.decomposition.DictionaryLearning(
        #    n_components=128, alpha=1, n_jobs=-1, tol=1e-4, verbose=False, max_iter=1000)
        self.dict_learner = sklearn.decomposition.DictionaryLearning(
            n_components=128, alpha=1, n_jobs=-1, transform_max_iter=100)

        if load is not None:
            self.load(load)

    def load(self, ckpt):
        '''
        Load checkpoint.
        '''
        array = np.load(ckpt, allow_pickle=False)
        self.dict_learner.components_ = array.copy()

    def save(self, ckpt):
        '''
        Save checkpoint.
        '''
        f = open(ckpt, 'wb')
        np.save(f, self.dict_learner.components_, allow_pickle=False)
        f.close()

    def train(self, images):
        '''
        images: training data, can be either
            paths for images (TBC),
            or list of TopBed (TBC),
            or numpy array with shape [N, 1, C, H].
        '''
        for _ in range(self.dict_learner.n_iter):
            print("images.shape",images.shape)
            X = [extract_patches_2d(image, self.patch_size, max_patches=10) for image in images]
            X = np.concatenate(X, axis=0)
            X = X.reshape(X.shape[0], -1)
            self.dict_learner.partial_fit(X)
            #print(self.dict_learner.n_iter_)
            print(self.dict_learner.iter_offset_)
        # image.PatchExtractor
        # numpy.lib.stride_tricks.sliding_window_view

    def eval2(self, image, bbox=False):
        X = extract_patches_2d(image, self.patch_size)
        X = X.reshape(X.shape[0], -1)
        X_transformed = self.dict_learner.transform(X)
        X_hat = X_transformed @ self.dict_learner.components_
        X_hat = X_hat.reshape(len(X_hat), *self.patch_size)
        recon = reconstruct_from_patches_2d(X_hat, image.shape)
        error = recon - image
        print("error.shape",error.shape)
        if bbox:
            bboxes = []
            error = error_to_bin(error)
            label, n_features = scipy.ndimage.label(error)
            for cnt in range(1, n_features+1):
                y, x = np.nonzero(label == cnt)
                bboxes.append((x.min()-1, y.min()-1, x.max()+1, y.max()+1))
            return bboxes
        else:
            return error

    def eval(self, image, bbox=False):
        X = extract_patches_2d(image, self.patch_size)
        X = X.reshape(X.shape[0], -1)
        X_transformed = self.dict_learner.transform(X)
        X_hat = X_transformed @ self.dict_learner.components_
        X_hat = X_hat.reshape(len(X_hat), *self.patch_size)
        recon = reconstruct_from_patches_2d(X_hat, image.shape)
        error = recon - image

        if bbox:
            print('bbox')
            bboxes = []
            error = error_to_bin(error)
            #save_path=r'C:\Users\Z004F4FY\Desktop\bed-ot\error.png'
            #save_path2 = r'C:\Users\Z004F4FY\Desktop\bed-ot\error_morph.png'
            #
            #imageio.imsave(save_path,np.clip(error * 255, 0, 255).astype(np.uint8))
            #imageio.imsave(save_path2,np.clip(error * 255 / 10, 0, 255).astype(np.uint8))
#            img = PIL.Image.fromarray(error).convert('RGB')
#            img.save(save_path)
            #
            label, n_features = scipy.ndimage.label(error)
            area=0
            mask_error = np.zeros(shape=error.shape)
            for cnt in range(1, n_features+1):
                y, x = np.nonzero(label == cnt)
                area_label=len(y)
                if area_label>=area:
                    area=area_label
                    y1, x1=y, x
            mask_error[y1, x1]=255

            height, width = mask_error.shape

            # 声明变换矩阵 参数1向右平移50个像素， 参数2向下平移100个像素
            M = np.float32([[1, 0, 100], [0, 1, 200]])
            # 进行2D 仿射变换
            shifted = cv2.warpAffine(mask_error, M, (width, height))

            save_path = r'C:\Users\Z004F4FY\Desktop\bed-ot\error_image.png'
            imageio.imsave(save_path, np.clip(mask_error * 255, 0, 255).astype(np.uint8))

            save_path1 = r'C:\Users\Z004F4FY\Desktop\bed-ot\shifted_image.png'
            imageio.imsave(save_path1, np.clip(shifted * 255, 0, 255).astype(np.uint8))

            image_made=image
            image_made[np.nonzero(shifted)]=image_made[np.nonzero(mask_error)]
            save_path2 = r'C:\Users\Z004F4FY\Desktop\bed-ot\made_image.png'
            imageio.imsave(save_path2,image_made)

            image_2=image
            mask_defect= np.zeros(shape=error.shape)
            mask_defect[np.nonzero(mask_error)]=image_2[np.nonzero(mask_error)]
            save_path3 = r'C:\Users\Z004F4FY\Desktop\bed-ot\mask_defect.png'
            imageio.imsave(save_path3,mask_defect)

            #bboxes.append((x.min()-1, y.min()-1, x.max()+1, y.max()+1))

            bboxe = [x1.min() - 1, y1.min() - 1, x1.max() + 1, y1.max() + 1]
            bboxes.append(bboxe)
            return bboxes
        else:
            return error

def main(args):
    if args.train is not None:
        # process data
        print('preparing data...')
        images = load_data(args.data)
        images = np.array(images)
        # prepare model
        print('preparing model...')
        model = DetectionAsDict()
        print('training...')
        model.train(images)
        model.save(args.train)
    else:
        # process data
        print('preparing data...')
        topbed = TopBed(os.path.join(args.data,'dicom'))
        # prepare model
        print('loading model...')
        assert args.eval is not None
        model = DetectionAsDict(load=args.eval)
        print('evaluation...')
        image = topbed.get_DX_intensity() 
        error = model.eval(image.copy())
        imageio.imsave('error.png', \
            np.clip(np.abs(error)*255/10, 0, 255).astype(np.uint8))
        error = scipy.ndimage.filters.convolve(error, np.array([[0,1,0],[1,1,1],[0,1,0]])/5)#(2,2))#np.ones((2,2)))
        error = np.abs(error)
        #error = skimage.morphology.erosion(error, skimage.morphology.square(2))
        #error = skimage.morphology.dilation(erGror, skimage.morphology.square(2))
        #fig = plt.figure()
        #plt.imshow(image, cmap='gray')
        #fig = plt.figure()
        #plt.imshow(error)
        topbed.save_as_image('image.png')
        imageio.imsave('error_morph.png', \
            np.clip(error*255/10, 0, 255).astype(np.uint8))
        imageio.imsave('error_bin.png', \
            np.clip((error>1.5)*255, 0, 255).astype(np.uint8))
        print('Done.')


def error_to_bin(error):
    error = scipy.ndimage.filters.convolve(
        error,
        np.array([[0,1,0],[1,1,1],[0,1,0]])/5)
    error = np.abs(error)
    error = error > 1.5
    #error = scipy.ndimage.morphology.binary_dilation(error, np.ones((5,5)))
    return error

def load_data(csv_file):
    f = open(csv_file, 'rt')
    images = []
    for one_path in f:
        one_path = one_path.strip()
        one_path = os.path.join(os.path.dirname(csv_file), one_path)
        one_file = TopBed(one_path)
        images.append(one_file.get_DX_intensity())
    return images

#     imgPath = imgDir+'/dicom'
#     img = load(imgPath)
#     shape = img.shape
# 
#     if recon_edge:
#         imageio.imsave(imgDir+'/image.png', \
#             np.clip(img*255, 0, 255).astype(np.uint8))
#         edge = sobel(img)
#         imageio.imsave(imgDir+'/target.png', \
#             np.clip(edge*255, 0, 255).astype(np.uint8))
#         target = edge
#     else:
#         imageio.imsave(imgDir+'/target.png', \
#             np.clip(img*255, 0, 255).astype(np.uint8))
#         target = img
# 
#     print('fitting...')
#     stamp = time.time()
#     print('done in {:.1f} seconds'.format(time.time()-stamp))
#     print('transforming...')
#     stamp = time.time()
# 
#     imageio.imsave(imgDir+'/recon.png', \
#         np.clip(recon*255, 0, 255).astype(np.uint8))
# 
#     error_map = np.abs(target - recon)
#     '''
#     error_map = np.max(np.abs(X_hat - X), axis=-1)
#     error_map = error_map.reshape(np.array(shape)-patch_size+1)
#     error_map = np.pad(error_map, [[patch_size//2]*2]*2)
#     '''
#     imageio.imsave(imgDir+'/error.png', \
#         np.clip(error_map*255*4, 0, 255).astype(np.uint8))
# 
#     #print(error_map.max())
#     imageio.imsave(imgDir+'/detect.png', \
#         ((error_map>=(5/WW))*255).astype(np.uint8))
#     #    ((error_map>=(10/WW))*255).astype(np.uint8))
# 
#         print('done in {:.1f} seconds'.format(time.time()-stamp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dictionary Learning for Defect Detection')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', type=str, default=None, metavar='/path/to/ckpt', \
        help='Path to checkpoint that will be saved.')
    group.add_argument('--eval', type=str, default=None, metavar='/path/to/ckpt', \
        help='Path to checkpoint that will be loaded.')
    parser.add_argument('--data', type=str, required=True, metavar='/path/to/data', \
        help='Index of data to be evaluated or trained.')
    args = parser.parse_args()
    main(args)
