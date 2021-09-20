"""
An Example of classification with tensorflow.

Datasets: mnist and fashion mnist
API: tensorflow sequential API
     tensorboard and call back
"""

# External module import
import os
import warnings
import datetime
from loguru import logger
import numpy as np
import sklearn
from typing import Any, Dict
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# Local module import

# Supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')



def dense_model(hyperparams_dense: dict):
    """Returns a sequential model

    Args:
        hyperparams_dense: model hyperparameters

    Returns:
        [type]: [description]
    """
    model_name = 'dense_layers_model'
    model = Sequential([
        #Flatten(input_shape=hyperparams_dense['input_shape']), 
        Dense(units=hyperparams_dense['units_layer_1'], \
              activation='relu', name='layer_1', input_shape=(28*28, )),
        BatchNormalization(name='batch_norm_1'),
        Dense(units=hyperparams_dense['units_layer_2'], \
              activation='relu', name='layer_2'),
        BatchNormalization(name='batch_norm_2'),
        Dense(units=hyperparams_dense['units_last_layer'], \
              activation='softmax', name='last_layer')
    ], name=model_name)
    return model, model_name


def cnn_model():
    """Returns a convolution neural net model.

    Args:
        hyperparams_cnn: model hyperparameters
    """
    ...
    model_name = 'convolution_network_model'
    model = Sequential([
        Rescaling(1./127.5, offset=-1, name='rescaling_0_1'),
        Conv2D(input_shape=(28, 28, 1),
               kernel_size=(3, 3),
               filters=128,
               strides=2,
               padding='same', 
               activation='relu', name='conv2d_layer_1'),
       BatchNormalization(name='batch_norm_1'),
       #MaxPooling2D(name='max_pool_1'),
       Conv2D(input_shape=(28, 28, 1),
               kernel_size=(3, 3),
               filters=64,
               strides=2,
               padding='same', 
               activation='relu', name='conv2d_layer_2'),
        BatchNormalization(name='batch_norm_2'),
        #MaxPooling2D(name='max_pool_2'),
        Conv2D(input_shape=(28, 28, 1),
               kernel_size=(3, 3),
               filters=32,
               strides=2,
               padding='same', 
               activation='relu', name='conv2d_layer_3'),
        BatchNormalization(name='batch_norm_3'),
        Flatten(),
        Dense(units=10, activation='softmax', name='output_layer')

    ], name=model_name) 
    return model, model_name



if __name__ == '__main__':
    ...

