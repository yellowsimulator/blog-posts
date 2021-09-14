"""
This files contains parameters such as
directory names, models hyperparameters ...
"""

from datetime import datetime
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# tensorboard log directory
log_dir = "logs/fit/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# hyperparameters
hyperparams_dense = {
        'input_shape': (28, 28),
        'units_layer_1': 512,
        'units_layer_2': 512,
        'units_last_layer': 10,
        'epochs': 2,
        'optimizer': 'adam',
        'metric': 'accuracy',
        'loss': SparseCategoricalCrossentropy()
    }

class_name_fashion_mnist = [
             'T-shirt/top', 'Trouser', 
             'Pullover', 'Dress', 'Coat',
             'Sandal', 'Shirt', 'Sneaker', 
             'Bag', 'Ankle boot']

def label_name(label_id):
    label_map = {
        k:class_name_fashion_mnist[k]
        for k in range(len(class_name_fashion_mnist))
    }
    return label_map[label_id]

    