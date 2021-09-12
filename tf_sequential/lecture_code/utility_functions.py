"""
A file containing utilities functions
"""
import io
import os
import cv2
import itertools
import sklearn
import numpy as np
import tensorflow as tf
from typing import List
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


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



# def log_confusion_matrix(test_X: np.ndarray, test_y: np.ndarray, 
#                          model: tf.keras.models, epoch: int, 
#                          log_dir: str, class_names: list):
#     """Logs a confusion matrix to TensorBoard.

#     Args:
#         test_X: Input test data fetures.
#         test_y: Input data test labels/target.
#         model: The trained model.
#         epoch: Training epoch.
#         log_dir: The directory containing tensorflow artifacts.
#         class_names: The class names or labels.
#     """
#     test_pred_proba = model.predict(test_X)
#     test_pred_label = np.argmax(test_pred_proba, axis=1)
#     confusion_matrix = sklearn.metrics.confusion_matrix(test_y, \
#                                           test_pred_label)
#     cm_figure = get_confusion_matrix_plot(confusion_matrix, \
#                                           class_names)
#     cm_image = plt_plot_to_tf_image(cm_figure)
#     cm_writer = tf.summary.create_file_writer(log_dir + \
#                                      '/confusion_metrix')
#     with cm_writer.as_default():
#         tf.summary.image("Confusion Matrix", cm_image, step=epoch)



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
   
