"""
This files contains parameters such as
directory names, models hyperparameters ...
"""

from datetime import datetime
from tensorflow.keras.datasets import fashion_mnist

# tensorboard log directory
log_dir = "logs/fit/" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# hyperparameters
hyperparams = {
        'input_shape': (28, 28),
        'units_layer_1': 512,
        'units_layer_2': 512,
        'units_last_layer': 10,
        'epochs': 100,
        'optimizer': 'adam',
        'metric': 'accuracy'
    }

class_name_fashion_mnist = ['T-shirt/top', 'Trouser', 
             'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 
            'Bag', 'Ankle boot']

    