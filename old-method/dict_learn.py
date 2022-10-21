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

from utils_io import TopBed

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
            print(images.shape)
            X = [extract_patches_2d(image, self.patch_size, max_patches=10) for image in images]
            X = np.concatenate(X, axis=0)
            print(X.shape)
            X = X.reshape(X.shape[0], -1)
            self.dict_learner.partial_fit(X)
            #print(self.dict_learner.n_iter_)
            print(self.dict_learner.iter_offset_)
        # image.PatchExtractor
        # numpy.lib.stride_tricks.sliding_window_view

    def eval(self, image, bbox=False):
        X = extract_patches_2d(image, self.patch_size)
        X = X.reshape(X.shape[0], -1)
        X_transformed = self.dict_learner.transform(X)
        X_hat = X_transformed @ self.dict_learner.components_
        X_hat = X_hat.reshape(len(X_hat), *self.patch_size)
        recon = reconstruct_from_patches_2d(X_hat, image.shape)
        error = recon - image
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
    error = scipy.ndimage.morphology.binary_dilation(error, np.ones((5,5)))
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
