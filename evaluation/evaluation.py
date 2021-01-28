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
from skimage.transform import resize
from ndb import *
import pickle
from pathlib import Path

from ndb import *

# # Read in folder containing generated images
# def read_file(img_dir_path):
#     folder_path = img_dir_path
#     imgs = []
#     for filename in glob.glob(os.path.join(folder_path, '*.png')):
#         img = cv2.imread(filename)
#         imgs.append(img)
#         print(filename)
#     return imgs
#
# def create_img_batch(imgs):
#     return np.stack(tuple(imgs))


def calculate_inception_score(images, n_split=5, eps=1E-16):
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

def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


# calculate frechet inception distance
def calculate_fid(images1, images2):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(96, 384, 3))
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
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
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
    #     img_folder = './generated_images/'
    #     imgs = read_file(img_folder)
    #     images = create_img_batch(imgs)
    objects = []
    origin_path = Path("./")
    for filename in origin_path.glob("*.pkl"):
        with (open(filename, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
    with (open('./generated_images/generated_images.pkl', "rb")) as openfile:
        test_images = pickle.load(openfile)

    train_images = []
    train_labels = []
    width = 64
    height = 32
    EPOCHS = 10
    num_examples_to_generate = 1

    for inner_layer in objects:
        label = inner_layer[0]
        for i in range(1, len(inner_layer)):
            for soundfile in inner_layer[i]:
                num_images = len(soundfile[0]) // width
                for j in range(num_images):
                    index1 = j * width
                    index2 = (j + 1) * width
                    train_images.append(soundfile[0:width, index1:index2])
                    train_labels.append(label)
    train_images = np.array(train_images)
    #train_labels = np.array(train_labels)
    # parse data to floats:
    train_images = train_images.reshape(train_images.shape[0], height, width, 1).astype('float32')
    print('Done')
    train_images_ = scale_images(train_images[:10000], (96, 384, 3))
    # test_images = scale_images(images, (96, 384, 3))
    # print(train_images_.shape)
    #print(test_images.shape)
    # train_images = np.stack((train_images,)*3, axis=-1)
    # print(train_images.shape)
    # train_samples = create_img_batch(train_images)
    # print(train_samples.shape)
    # calculate inception score
    is_avg, is_std = calculate_inception_score(train_images_)
    is_avg_t, is_std_t = calculate_inception_score(test_images)
    print('TRAIN Inception score avg:', is_avg, "Inception score std:", is_std)
    print('TEST Inception score avg:', is_avg_t, "Inception score std:", is_std_t)
    print("---------------")
    fid = calculate_fid(train_images_, test_images)
    print('FID score: %.3f' % fid)

    batch = []
    for i in train_images_:
        batch.append(i.ravel())
    images_ndb = np.array(batch)

    for i in test_images:
        batch.append(i.ravel())
    test_images = np.array(batch)

    num_bins = 5
    #     n_train = round(images_ndb.shape[0]*0.7)

    #     rand_order = np.random.permutation(images_ndb.shape[0])
    #     train_samples = images_ndb[rand_order[:n_train]]
    #     test_samples = images_ndb[rand_order[n_train:]]

    calculate_bins(images_ndb, test_images, num_bins)

if __name__ == '__main__':
    main()

