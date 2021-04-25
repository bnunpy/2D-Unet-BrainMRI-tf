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
        'num_channels': tf.io.FixedLenFeature((), tf.int64),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)/255
    image_shape = [example['height'], example['width'], example['num_channels']]
    image = tf.reshape(image, image_shape)
    #image = tf.squeeze(image)

    mask = tf.io.parse_tensor(example['mask'], out_type=tf.uint8)
    mask_shape = [example['height'], example['width']]
    mask = tf.reshape(mask, mask_shape)
    #mask = tf.squeeze(mask)
    #mask = mask[tf.newaxis, ...]
    #mask = tf.stack((mask,mask,mask,mask))
    #mask = tf.reshape(mask, (224,224,4))

    #mask = tf.cast(mask, tf.float32)
    mask = tf.one_hot(mask, params.NUM_CLASSES)
    #mask = tf.squeeze(mask)
    print('im', image.shape, 'msk', mask.shape)
    return image, mask


def get_tfrecords_dataset(mode):
    if mode == 'training':
        tfrecords_dir=params.C_TRAIN_TFRECORDS_PATH
    elif mode == 'validation':
        tfrecords_dir=params.C_VALID_TFRECORDS_PATH
    elif mode == 'evaluation':
        tfrecords_dir=params.C_EVAL_TFRECORDS_PATH
    else:
        raise('Error: Invalid mode selected.')

    filenames = glob.glob(f'{tfrecords_dir}\*')
    tfrecord_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    return tfrecord_dataset.map(read_tfrecord)

#dataset = get_tfrecords_dataset('training').range(1)
#print(list(dataset.as_numpy_iterator()))


def prepare_dataset(dataset, batch_size, buffer_size=200, repeat=0, take=0, cache=False, shuffle=True, prefetch=True):

    if take > 0:
        dataset = dataset.take(take)

    if shuffle:  # shuffle before repeat to have the maximum randomness
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    if repeat > 0:
        dataset = dataset.repeat(repeat)
    elif repeat == -1:
        dataset = dataset
    else:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    if cache:  # cache after batching
        dataset = dataset.cache()

    if prefetch:  # prefetch at the end
        dataset = dataset.prefetch(buffer_size // batch_size)

    return dataset