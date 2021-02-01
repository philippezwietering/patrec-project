#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os
import glob
import cv2
import numpy as np     
from scipy.linalg import sqrtm
from math import floor

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from skimage.transform import resize
from ndb import *
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[36]:


def simple_cnn(width):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# In[37]:


def train_model(width, model):
    objects = []
    origin_path = Path("./")
    for filename in origin_path.glob("*.pkl"):
        with (open(filename, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break

#     train_images = []
    train_labels = []
    width = width
    height = 32

    for inner_layer in objects:
        label = inner_layer[0]
        for i in range(1, len(inner_layer)):
            for soundfile in inner_layer[i]:
                num_images = len(soundfile[0]) // width
                for j in range(num_images):
                    index1 = j*width
                    index2 = (j+1)*width
#                     train_images.append(soundfile[0:width, index1:index2])
                    train_labels.append(label)  
#     images = np.array(train_images)
    
    
#     train_images = scale_images(images, (32, width, 3)) # Scale with RBG Channel

#     with open('training_data/scaled_train_images_'+str(width)+'.pkl', 'wb') as f: # save the stuff
#         pickle.dump(train_images, f)

    with open('training_data/scaled_training_data'+str(width)+'.pkl', 'rb') as f:
        train_images = pickle.load(f)
    train_labels = np.array(train_labels)
    label_encoder = LabelEncoder()
    train_labels = np.array(label_encoder.fit_transform(train_labels))
    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.33, random_state=42)
    
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    model.save('trained_model'+str(width)+'.h5')
    
    plt.figure()  
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('Accuracy_cnn'+str(width)+'.png')
    plt.show()


    test_loss, test_acc = model.evaluate(X_test,  y_test)
    print('Model'+str(width) + 'test accuracy:', test_acc)
    print('Model'+str(width)+'test_loss:', test_loss)
    
    


# In[38]:


# model128 = simple_cnn(128)
# model512 = simple_cnn(512)
# train_model(128, model128)
# train_model(512, model512)


# In[39]:


def calculate_inception_score(model, images, n_split=5, eps=1E-16):
        # predict class probabilities for images
        yhat = model.predict(images)
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


# In[40]:


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
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


# In[41]:


def calculate_bins(train_samples, test_samples, k, fname):
    ndb = NDB(training_data=train_samples, number_of_bins=k)
    ndb.evaluate(test_samples, model_label='Test')
    plt.figure()
    ndb.plot_results()
    plt.savefig(fname+'bins'+str(k)+'.png')
    plt.show()


# In[42]:


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


# In[43]:


# Resize shapes
def bin_preprocessing(images):
    batch = []
    for i in images:
        batch.append(i.ravel())
    return np.array(batch)


# In[47]:


def main():
    
    with (open('./generated_images/generated_imgs_128_niet_upsampled.pkl', "rb")) as openfile:
        test_images128 = scale_images(pickle.load(openfile), (32, 128, 3))
    with (open('./generated_images/generated_imgs_512_niet_upsampled.pkl', "rb")) as openfile:
        test_images512 = scale_images(pickle.load(openfile), (32, 512, 3))
    with (open('./training_data/scaled_training_data128.pkl', "rb")) as openfile:
        train_images128 = pickle.load(openfile)
    with (open('./training_data/scaled_training_data512.pkl', "rb")) as openfile:
        train_images512 = pickle.load(openfile)
        

    model128_fc = keras.models.load_model('trained_model128.h5')   
    model512_fc = keras.models.load_model('trained_model512.h5') 
    
    
    # Remove classification layers for features
    model128 = models.Sequential(model128_fc.layers[:-1])
    model512 = models.Sequential(model512_fc.layers[:-1])    
    
    X_train128, X_val128 = train_test_split(train_images128, test_size=0.33, random_state=42)
    X_train512, X_val512 = train_test_split(train_images512, test_size=0.33, random_state=42)
     
    is_avg128, is_std128 = calculate_inception_score(model128_fc, X_val128[:2000]) # TRAIN128 1.4650955 Inception score std: 0.004975435
    is_avg_t128, is_std_t128 = calculate_inception_score(model128_fc, test_images128) # TEST128 Inception score avg: 1.5592765 Inception score std: 0.014011001
    is_avg512, is_std512 = calculate_inception_score(model512_fc, X_val512[:200]) # TRAIN512 Inception score avg: 1.8127365 Inception score std: 0.018795773
    is_avg_t512, is_std_t512 = calculate_inception_score(model512_fc, test_images512) # TEST512 Inception score avg: 1.000068 Inception score std: 0.0001033097

    print('TRAIN128 Inception score avg:', is_avg128, "Inception score std:", is_std128)
    print('TEST128 Inception score avg:', is_avg_t128, "Inception score std:", is_std_t128)
    print('TRAIN512 Inception score avg:', is_avg512, "Inception score std:", is_std512)
    print('TEST512 Inception score avg:', is_avg_t512, "Inception score std:", is_std_t512)
    print("---------------")
    
    fid128X = calculate_fid(model128, X_train128, X_val128[:2000]) #FIDX score 128: 0.227
    print('FID score 128X: %.3f' % fid128X)
    fid512X = calculate_fid(model512, X_train512, X_val512[:200]) # FIDX score 512: 6.098
    print('FID score 512X: %.3f' % fid512X)
    
#     fid128 = calculate_fid(model128, train_images128, test_images128) #FID score 128: 36.925
#     print('FID score 128: %.3f' % fid128)
#     fid512 = calculate_fid(model512, train_images512, test_images512) # FID score 512: 248.642
#     print('FID score 512: %.3f' % fid512)
#     print("---------------")

    # Calculate NDB
    X_train_bins128 = bin_preprocessing(X_train128)
    X_test_bins128 = bin_preprocessing(X_val128[:2000])   
    
    X_train_bins512 = bin_preprocessing(X_train512)
    X_test_bins512 = bin_preprocessing(X_val512[:200])    
    
    calculate_bins(X_train_bins128, X_test_bins128, 10, 'Xndb128') # NDB = 0 NDB/K = 0.0 , JS = 0.00030114696948060665
    calculate_bins(X_train_bins512, X_test_bins512, 10, 'Xndb512') # NDB = 0 NDB/K = 0.0 , JS = 0.002305073461703629
    
    calculate_bins(X_train_bins128, X_test_bins128, 50, 'Xndb128') # NDB = 3 NDB/K = 0.06 , JS = 0.0032261136221497814
    calculate_bins(X_train_bins512, X_test_bins512, 50, 'Xndb512') # NDB = 1 NDB/K = 0.02 , JS = 0.0065002510446860555
    
    calculate_bins(X_train_bins128, X_test_bins128, 100, 'Xndb128') # NDB = 2 NDB/K = 0.02 , JS = 0.005063471466338541
    calculate_bins(X_train_bins512, X_test_bins512, 100, 'Xndb512') # NDB = 3 NDB/K = 0.03 , JS = 0.02544914328090541 
    
    calculate_bins(X_train_bins128, X_test_bins128, 200, 'Xndb128') # NDB = 8 NDB/K = 0.04 , JS = 0.009228856394977109
    calculate_bins(X_train_bins512, X_test_bins512, 200, 'Xndb512') # NDB = 3 NDB/K = 0.015 , JS = 0.03345394759694796   
         
    
    train_bins128 = bin_preprocessing(train_images128)
    test_bins128 = bin_preprocessing(test_images128) # 2000 generated samples
    train_bins512 = bin_preprocessing(train_images512)
    test_bins512 = bin_preprocessing(test_images512) # 200 generated samples
    
    calculate_bins(train_bins128, test_bins128, 10, 'ndb128') # Results for 2000 samples from Test: NDB = 9 NDB/K = 0.9 , JS = 0.014024162216545719
    calculate_bins(train_bins512, test_bins512, 10, 'ndb512') # Results for 200 samples from Test: NDB = 1 NDB/K = 0.1 , JS = 0.017794418726491
    
    calculate_bins(train_bins128, test_bins128, 50, 'ndb128') # Results for 2000 samples from Test: NDB = 18 NDB/K = 0.36 , JS = 0.11688896580014627
    calculate_bins(train_bins512, test_bins512, 50, 'ndb512') # Results for 200 samples from Test: NDB = 2 NDB/K = 0.04 , JS = 0.04331794883881373
    
    calculate_bins(train_bins128, test_bins128, 100, 'ndb128') # Results for 2000 samples from Test: NDB = 21 NDB/K = 0.21 , JS = 0.13734317154549774
    calculate_bins(train_bins512, test_bins512, 100, 'ndb512') # Results for 200 samples from Test: NDB = 3 NDB/K = 0.03 , JS = 0.06817458863922093   
    
    calculate_bins(train_bins128, test_bins128, 200, 'ndb128') # Results for 2000 samples from Test: NDB = 29 NDB/K = 0.145 , JS = 0.1978833951840465
    calculate_bins(train_bins512, test_bins512, 200, 'ndb512') # Results for 200 samples from Test: NDB = 5 NDB/K = 0.025 , JS = 0.09931714987806105   
       


# In[48]:


if __name__ == '__main__':
    main()


# In[ ]:




