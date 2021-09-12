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
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
#from tensorflow.python.keras.backend import flatten
# Local module import
from params import log_dir, hyperparams, class_name_fashion_mnist
from utility_functions import get_confusion_matrix_plot
from utility_functions import plt_plot_to_tf_image
# Supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


@logger.catch
def get_data(data_name: str='mnist'):
    """Returns train and test data. 

    Args:
        data_name: name of the dataset.
                  'mnist' or 'fashion_mnist'

    Returns:
        [tuple, np.ndarray]: train and test data
    """
    data = {'mnist': mnist, 
            'fashion_mnist': fashion_mnist}
    (train_X, train_y), (test_X, test_y) = data[data_name].load_data()
    return train_X, train_y, test_X, test_y



@logger.catch
def get_model(hyperparams: Dict):
    """Returns a sequential model

    Args:
        hyperparams ([type]): [description]

    Returns:
        [type]: [description]
    """
    return Sequential([
        Flatten(input_shape=hyperparams['input_shape']), 
        Dense(units=hyperparams['units_layer_1'], \
              activation='relu', name='layer_1'),
        BatchNormalization(),
        Dense(units=hyperparams['units_layer_2'], \
              activation='relu', name='layer_2'),
        BatchNormalization(),
        Dense(units=hyperparams['units_last_layer'], \
              activation='softmax', name='last_layer')
    ])


@logger.catch
def train_model(class_names: str, hyperparams: Dict,\
                data_name: str='mnist'):
    """Trains a sequential model.

    Args:
        model: our sequential model.
        hyperparams: dictionary of hyperparameters
        data_name: dataset name. Defaults to 'mnist'.
    """
    model = get_model(hyperparams)
    train_X, train_y, test_X, test_y = get_data(data_name=data_name)
    # 1 - compile the model
    model.compile(
        loss=SparseCategoricalCrossentropy(), 
        optimizer=hyperparams['optimizer'],
        metrics=[hyperparams['metric']]
    )
    # 2 - fit and evaluate model
    cm_writer = tf.summary.create_file_writer(log_dir + '/cm')
    def log_confusion_matrix(epoch, log_dir):
        test_pred_proba = model.predict(test_X)
        test_pred_label = np.argmax(test_pred_proba, axis=1)
        confusion_matrix = sklearn.metrics.confusion_matrix(test_y, \
                                          test_pred_label)
        cm_figure = get_confusion_matrix_plot(confusion_matrix, \
                                          class_names)
        cm_image = plt_plot_to_tf_image(cm_figure)
        with cm_writer.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    
    tensorboar_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    confusion_matrix_callback = LambdaCallback(on_epoch_end=log_confusion_matrix)
    model.fit(x=train_X,
              y=train_y, 
              epochs=hyperparams['epochs'],
              validation_data=(test_X, test_y),
              callbacks=[tensorboar_callback, confusion_matrix_callback])
    


def run_training(data_name='mnist'):
    """Runs training."""
    label_names = {
        'mnist': [k for k in range(10)],
        'fashion_mnist': class_name_fashion_mnist
    }
    class_names = label_names[data_name]
    train_model(class_names, hyperparams, data_name)


# @logger.catch
# def tune_hyperparams():
#     # perform hyperparameters 
#     # testing
#     pass


# @logger.catch
# def deploy_model():
#     # Deloy the model in production
#     pass


# @logger.catch
# def test_hyperparams():
#     # perform hypothese testing
#     # for different hyperparameter



if __name__ == '__main__':
    run_training(data_name='fashion_mnist')

