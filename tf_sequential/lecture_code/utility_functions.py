"""
A file containing utilities functions
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import io
import itertools
import matplotlib.pyplot as plt


def plt_plot_to_tf_image(figure: plt.figure(figsize=(10, 10))):
    """Comverts a matplotlib plot to tensorflow image.

    Credit: 
        https://www.tensorflow.org/tensorboard/image_summaries

    Args:
        figure: matplotlib figure.
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format='PNG')
    plt.close(figure)
    buffer.seek(0)
    tf_image = tf.image.decode_png(contents=buffer.getvalue(), channels=0)
    tf_image = tf.expand_dims(tf_image, axis=0)
    return tf_image


def plot_images(nb_images: int, X: np.ndarray,
                y: np.ndarray, plot_image=True):
    """plots images from a classification task.

    Args:
        nb_images: number of images to plot.
        X: image array.
        y: label array.
    """
    figure = plt.figure(figsize=(10,10))
    factors = [k if nb_images%k==0 else -1 \
                 for k in range(1, nb_images + 1)]
    w, h = factors[0], factors[1]
    if len(factors) != 2:
        w, h = 5, 5
        nb_images = 25
    for i in range(nb_images):
        plt.subplot(w,h,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i], cmap=plt.cm.binary)
        plt.xlabel(y[i])
    if plot_image is True:
        plt.show()
    return figure


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Credit: 
        https://www.tensorflow.org/tensorboard/image_summaries

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure




# # Define the per-epoch callback.
# cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
