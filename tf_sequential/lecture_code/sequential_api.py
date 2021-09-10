# External module import
import os
import warnings
import datetime
import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# Local module import
from utility_functions import plot_images
from utility_functions import plt_plot_to_tf_image
from utility_functions import plot_confusion_matrix
# Supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')



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
    (train_X, train_y), (test_X, test_y) = \
                 data['data_name'].load_data()
    return train_X, train_y, test_X, test_y




def build_model():
    # build the first model
    pass


def tune_hyperparams():
    # perform hyperparameters 
    # testing
    pass


def deploy_model():
    # Deloy the model in production
    pass


def test_hyperparams():
    # perform hypothese testing
    # for different hyperparameter





