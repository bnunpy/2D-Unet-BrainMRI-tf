import os
import sys
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import glob
import IPython.display as display

from configure import params


@tf.function
def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),  # tf.float32, image between 0 and 1
        'mask': tf.io.FixedLenFeature((), tf.string),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)

    mask = tf.io.parse_tensor(example['mask'], out_type=tf.uint8)
    mask_shape = [example['height'], example['width']]
    mask = tf.reshape(mask, mask_shape)

    return image, mask


def get_tfrecords_dataset(mode):
    if mode == 'training':
        tfrecords_dir=params.TRAIN_TFRECORDS_PATH
    elif mode == 'validation':
        tfrecords_dir=params.VALID_TFRECORDS_PATH
    else:
        raise('Error: Invalid mode selected.')

    filenames = glob.glob(f'{tfrecords_dir}\*')
    tfrecord_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    return tfrecord_dataset.map(read_tfrecord)

dataset = get_tfrecords_dataset('training').range(1)
print(list(dataset.as_numpy_iterator()))

