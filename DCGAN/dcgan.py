import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

# TODO: reference original script

#print(tf.__version__) # check we indeed have tensorflow version 2.4

# MNIST has 60'000 training images, each of size 28x28
(train_data, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# replace above by loading our bird spectrograms of size 28x28
print(train_data.shape)
# parse data to floats:
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
# TODO: find min, max of our data to normalize to [-1,1]
train_data = (train_data - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256 # define a batch size

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
