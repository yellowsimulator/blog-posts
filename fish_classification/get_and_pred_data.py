"""
...
"""

import os
import sys
import pandas as pd
from glob import glob
import tensorflow as tf
sys.path.append(os.path.abspath('../'))
#from helper_functions.tf_callbacks import hello

PATH = '/Users/yapi/Desktop/DeepFish/Classification'
DATA_PATH = f'{PATH}/data'
METADATA_PATH = f'{PATH}/metadata'

TRAIN_PATH = f'{METADATA_PATH}/train.csv'
TEST_PATH = f'{METADATA_PATH}/test.csv'
VALID_PATH = f'{METADATA_PATH}/valid.csv'
print(METADATA_PATH)
METADATA = {
    'test': TEST_PATH,
    'train': TRAIN_PATH,
    'valid': VALID_PATH
}
test_metadata_df = pd.read_csv(TRAIN_PATH)
print(test_metadata_df)




