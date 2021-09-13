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
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
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
    model_name = 'dense layers model'
    model = Sequential([
        Flatten(input_shape=hyperparams_dense['input_shape']), 
        Dense(units=hyperparams_dense['units_layer_1'], \
              activation='relu', name='layer_1'),
        BatchNormalization(name='batch_norm_1'),
        Dense(units=hyperparams_dense['units_layer_2'], \
              activation='relu', name='layer_2'),
        BatchNormalization(name='batch_norm_2'),
        Dense(units=hyperparams_dense['units_last_layer'], \
              activation='softmax', name='last_layer')
    ], name=model_name)
    return model, model_name


def cnn_model(hyperparams_cnn: dict):
    """Returns a convolution neural net model.

    Args:
        hyperparams_cnn: model hyperparameters
    """
    model_name = 'convolution network layers model'
    model = Sequential([
        Conv2D(input_shape=(28, 28, 1),
               kernel_size=...,
               filters=...,
               strides=...,
               padding='same', 
               activation='relu', name='conv2d_layer_1'),
       BatchNormalization(name='batch_norm_1'),
       Conv2D(input_shape=(28, 28, 1),
               kernel_size=...,
               filters=...,
               strides=...,
               padding='same', 
               activation='relu', name='conv2d_layer_2'),
        BatchNormalization(name='batch_norm_2'),
        Flatten(input_shape=(28, 28, 1)),
        Dense(units=..., activation='softmax', name='output_layer')

    ], name=model_name) 
    return model, model_name



if __name__ == '__main__':
    ...

