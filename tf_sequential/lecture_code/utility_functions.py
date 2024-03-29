"""
A file containing utilities functions
"""

# External libraries
import io
import os
import cv2
import sklearn
import keras
import itertools
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import Image, display
import matplotlib.image as mpi
import matplotlib.cm as cm
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tf_explain.callbacks.grad_cam import GradCAMCallback
# Local libraries
from params import log_dir, hyperparams_dense
from params import  class_name_fashion_mnist



def plt_plot_to_tf_image(figure: plt.figure(figsize=(10, 10))):
    """Converts a matplotlib plot to tensorflow image.

    Credit: 
        https://www.tensorflow.org/tensorboard/image_summaries

    Args:
        figure: matplotlib figure.
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG')
    plt.close(figure)
    buffer.seek(0)
    tf_image = tf.image.decode_png(contents=buffer.getvalue(), \
           channels=0)
    tf_image = tf.expand_dims(tf_image, axis=0)
    return tf_image


def plot_images(rows: int, columns: int, 
                data_name: str, X: np.ndarray,
                y: np.ndarray, plot_image=True):
    """plots images from a classification task,
       by default plot 25 images.

    Args:
        nb_images: number of images to plot.
        X: image array.
        y: label array.
    """
    
    nb_images = rows*columns
    figure = plt.figure(figsize=(10,10))
    for i in range(nb_images):
        plt.subplot(rows,columns,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i], cmap=plt.cm.binary)
        if data_name == 'fashion_mnist':
            class_name = \
            ['T-shirt/top', 'Trouser', 
             'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 
            'Bag', 'Ankle boot']
            plt.xlabel(class_name[y[i]])
        else:
            plt.xlabel(y[i])
    if plot_image is True:
        plt.show()
    return figure


def get_confusion_matrix_plot(cm: np.ndarray, class_names: list):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Credit: 
        https://www.tensorflow.org/tensorboard/image_summaries

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): classe namse
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    #plt.show()
    return figure



def extract_frame_from_video(video_path: str, 
                             frame_format: str, 
                             frame_name: str,
                             target_path: str):
    """Extract images/frames from a video.

    Args:
        video_path: path to video
        frame_format: format of the extracted frame.
        frame_name: name of the extracted frame.
        target_path: path or directory name to save 
                     the frames.
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    frame_number = 0
    capture = cv2.VideoCapture(video_path)
    while(capture.isOpened()):
        frame_exists, frame = capture.read()
        if not frame_exists:
            break
        name = ''.join([frame_name, str(frame_number), \
                f'.{frame_format}'])
        path = f'{target_path}/{name}'
        cv2.imwrite(path, frame)
        frame_number += 1
    capture.release()
    cv2.destroyAllWindows()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """[summary]
    
    Creadit: https://keras.io/examples/vision/grad_cam/

    Args:
        img_array ([type]): [description]
        model ([type]): [description]
        last_conv_layer_name ([type]): [description]
        pred_index ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """[summary]

    Credit: https://keras.io/examples/vision/grad_cam/

    Args:
        img_path ([type]): [description]
        heatmap ([type]): [description]
        cam_path (str, optional): [description]. Defaults to "cam.jpg".
        alpha (float, optional): [description]. Defaults to 0.4.
    """
    img = img_path[0]#np.squeeze(img_path)
    if isinstance(img_path, str):
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :1]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    im = mpi.imread(cam_path)
    plt.imshow(im)
    plt.show()
    


if __name__ == '__main__':
    ...
    # videos_path = '../../video/inspection_camera'
    # video_files = glob(f'{videos_path}/*.mkv')
    # video_path = video_files[0]
    # name = os.path.basename(video_path).split('.')[0]
    # target_path = f'{videos_path}/{name}'
    # print(video_path)
    # print(name)

    # frame_format = 'jpg'
    # frame_name = 'frame'
    # extract_frame_from_video(video_path, frame_format, 
    #                           frame_name, target_path)

    # image = '../../video/inspection_camera/camera2_2020-11-25_11-46-48+0000/frame0.jpg'
   
