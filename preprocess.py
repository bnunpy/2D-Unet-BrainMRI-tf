import tensorflow as tf
import numpy as np
from configure import params


def resize_img(img, x_size=params.IMG_SIZE, y_size=params.IMG_SIZE):
    img_p = np.ndarray((x_size, y_size, params.NUM_CHANNELS), dtype=None)
    for i in range(params.NUM_CHANNELS):
        img_p[..., i] = np.resize(img[..., i], (x_size, y_size))
    return img_p


def resize_lbl(lbl, x_size=params.IMG_SIZE, y_size=params.IMG_SIZE):
    lbl_p = np.resize(lbl, (x_size, y_size))
    lbl_p = tf.one_hot(lbl_p, params.NUM_CLASSES)
    return lbl_p


def trim_edges(training_images, training_labels):
    '''Cuts zero edge for a stack of 2D images.
    Args:
        data: 2D image stack, [number_of_images, Height, Width,].
    Returns:
        stacks of trimmed images and labels
    '''

    x_size, y_size = training_images.shape[1:]
    for i in range(training_images.shape[0]):
        
        x_size_start, x_size_end = 0, x_size-1
        y_size_start, y_size_end = 0, y_size-1

        while training_images[i,:,y_size_start].sum() == 0:
            y_size_start += 1
        while training_images[i,:,y_size_end].sum() == 0:
            y_size_end -= 1
        while training_images[i,x_size_start,:].sum() != 0:
            x_size_start += 1
        while training_images[i,x_size_end,:].sum() != 0:
            x_size_end -= 1

        trimmed_training_images = training_images[x_size_start:x_size_end, y_size_start:y_size_end]
        trimmed_training_labels = training_labels[x_size_start:x_size_end, y_size_start:y_size_end]

    return (trimmed_training_images, trimmed_training_labels)