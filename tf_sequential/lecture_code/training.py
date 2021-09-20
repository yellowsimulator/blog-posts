# External libraries
import numpy as np
import tempfile
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import save_model
from tf_explain.callbacks.grad_cam import GradCAMCallback
# Local libraries
from params import log_dir, hyperparams_dense
from params import  class_name_fashion_mnist
from call_backs import ConfusionMatrixCallBack


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


def train_model(model: tf.keras.models,
                task_name: str,
                class_names: list,
                hyperparams: dict,
                data_name: str='mnist'):
    """Trains a sequential model.

    Args:
        model: our sequential model.
        task_name: the name of the task.
        class_names: class label 
        hyperparams: dictionary of hyperparameters
        data_name: dataset name. Defaults to 'mnist'.
    """
    MODEL_DIR = 'models_repo'
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))

    train_X, train_y, test_X, test_y = get_data(data_name=data_name)
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)
    # compile the model
    model.compile(
        loss=hyperparams['loss'], 
        optimizer=hyperparams['optimizer'],
        metrics=[hyperparams['metric']]
    )
    # call backs setup
    tensorboar_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss')
    csv_loger_callback = CSVLogger(filename='loger', append=True)
    confusion_matrix_callback = ConfusionMatrixCallBack(
        validation_data=(train_X, train_y),
        output_dir=f'logs/fit/confusion_matrix',
        class_names=class_names)
    grad_cam_callback = GradCAMCallback(
        validation_data=(test_X, test_y),
        layer_name='conv2d_layer_3',
        class_index=0,
        output_dir='logs/fit/grad_cam')
    # fit model
    model.fit(x=train_X,
              y=train_y, 
              epochs=hyperparams['epochs'],
              validation_data=(test_X, test_y),
              callbacks=[tensorboar_callback, 
                         confusion_matrix_callback,
                         early_stop_callback, 
                         grad_cam_callback])
    # save model
    model.save("model_repo/model1")

    
    

def run_training(model: tf.keras.models,
                 task_name: str, 
                 hyperparams: dict,
                 data_name='mnist'):
    """Runs training."""
    label_names = {
        'mnist': [k for k in range(10)],
        'fashion_mnist': class_name_fashion_mnist
    }
    class_names = label_names[data_name]
    train_model(model, task_name, class_names,
                 hyperparams, data_name)



if __name__ == '__main__':
    ...