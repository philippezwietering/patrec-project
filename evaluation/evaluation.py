#!/usr/bin/env python
# coding: utf-8

import os
import glob
import cv2
import numpy as np     
from scipy.linalg import sqrtm
from math import floor

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
# from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg19 import VGG19
# from keras.applications.vgg19 import preprocess_input

from ndb import *

# Read in folder containing generated images
def read_file(img_dir_path):
    folder_path = img_dir_path
    imgs = []
    for filename in glob.glob(os.path.join(folder_path, '*.png')):
        img = cv2.imread(filename)
        imgs.append(img)
        print(filename)
    return imgs  

def create_img_batch(imgs):
    return np.stack(tuple(imgs))


def calculate_inception_score(images, n_split=2, eps=1E-16):
        # load inception v3 model
        model = InceptionV3(include_top=False, weights='imagenet')
        
        # convert from uint8 to float32
        processed = images.astype('float32')
        # pre-process raw images for inception v3 model
        processed = preprocess_input(processed)
        # predict class probabilities for images
        yhat = model.predict(processed)
        # enumerate splits of images/predictions
        scores = list()
        n_part = floor(images.shape[0] / n_split)
        for i in range(n_split):
            # retrieve p(y|x)
            ix_start, ix_end = i * n_part, i * n_part + n_part
            p_yx = yhat[ix_start:ix_end]
            # calculate p(y)
            p_y = np.expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = np.mean(sum_kl_d)
            # undo the log
            is_score = np.exp(avg_kl_d)
            # store
            scores.append(is_score)
        # average across images
        is_avg, is_std = np.mean(scores), np.std(scores)
        return is_avg, is_std


# calculate frechet inception distance
def calculate_fid(images1, images2):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(400,400,3))
    images1 = preprocess_input(images1)    
    images2 = preprocess_input(images2)    
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# def sample_from(samples, number_to_use):
#     print(samples.shape)
#     print(number_to_use)
#     assert samples.shape[0] >= number_to_use
#     rand_order = np.random.permutation(samples.shape[0])
#     return samples[rand_order[:number_to_use], :]

# Visualize the missing bins
# def visualize_bins(bin_centers, is_different):
#     k = bin_centers.shape[0]
#     n_cols = 10
#     n_rows = (k+n_cols-1)//n_cols
#     for i in range(k):
#         plt.subplot(n_rows, n_cols, i+1)
#         plt.imshow(bin_centers[i, :].reshape([400, 400, 3]))
#         if is_different[i]:
#             plt.plot([0, 399], [0, 399], 'r', linewidth=2)
#         plt.axis('off')

def calculate_bins(train_samples, test_samples, k):
    ndb = NDB(training_data=train_samples, number_of_bins=k, whitening=True)
    ndb.evaluate(test_samples, model_label='Test')
    # ndb.plot_results(models_to_plot=['Test']) gives errors

def main():
    img_folder = './generated_images/'
    imgs = read_file(img_folder) 
    images = create_img_batch(imgs)

    # calculate inception score
    is_avg, is_std = calculate_inception_score(images)
    print('Inception score avg:', is_avg, "Inception score std:", is_std)
    print("---------------")
    # calculate fid
    fid = calculate_fid(images, images)
    print('FID score: %.3f' % fid)

    batch = []
    for i in imgs:
        batch.append(i.ravel())
    images_ndb = np.array(batch)

    num_bins = 2
    n_train = round(images_ndb.shape[0] * 0.7)

    rand_order = np.random.permutation(images_ndb.shape[0])
    train_samples = images_ndb[rand_order[:n_train]]
    test_samples = images_ndb[rand_order[n_train:]]

    # calculate NDB
    calculate_bins(train_samples, test_samples, num_bins)

if __name__ == '__main__':
    main()

