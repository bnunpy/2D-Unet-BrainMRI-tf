import os
import sys
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import glob
import nibabel as nib

from configure_3D import params

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Create a dictionary with features that may be relevant.
def serialize_example(image, mask, image_shape):

    feature = {
        'image': _bytes_feature(image),  # image dtype=float32
        'mask': _bytes_feature(mask),
        'num_classes': _int64_feature(params.NUM_CLASSES),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'num_channels': _int64_feature(image_shape[3])
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def select_patch(image, mask, patch_size):
    """
    Select a random patch on image, mask at the same location
    Args:
        image (tf.Tensor): Tensor for the input image of shape (h x w x (1 or 3)), dtype: float32
        mask (tf.Tensor): Tensor for the mask image of shape (h x w x 1), dtype: uint8
        patch_size (tuple): Size of patch (height, width)
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Tuple of tensors (image, mask) with shape (patch_size[0], patch_size[1], 3)
    """
    image = tf.image.convert_image_dtype(image, tf.uint8)
    mask = patch_mask = tf.expand_dims(mask, axis=-1)

    concat = tf.concat([image, mask], 
                axis=-1)
    patches = tf.image.random_crop(concat, 
                size=[patch_size[0], patch_size[1], patch_size[2], params.NUM_CHANNELS+1])
    patch_image = tf.image.convert_image_dtype(patches[:, :, :, :params.NUM_CHANNELS], tf.float32)
    patch_mask = tf.expand_dims(patches[:, :, :, -1], axis=-1)
    patch_mask = tf.squeeze(patch_mask)

    return (patch_image, patch_mask)


def data_generator(imgs_path, lbls_path):
    
    filenames = os.listdir(imgs_path)

    for filename in filenames:
        image_file = os.path.join(imgs_path, filename)
        label_file = os.path.join(lbls_path, filename)
        image3d = nib.load(image_file)        
        label3d = nib.load(label_file)
        image3d = np.asanyarray(image3d.dataobj, 
                    dtype=np.float32)[8:232,8:232,18:138,[params.CHANNELS_DICT[x] for x in params.CHANNELS]]
        label3d = np.asanyarray(label3d.dataobj, 
                    dtype=np.uint8)[8:232,8:232,18:138] # shape: (224,224,120)
        image3d = tf.convert_to_tensor(image3d, 
                    dtype=tf.float32)
        label3d = tf.convert_to_tensor(label3d, 
                    dtype=tf.uint8)

        for patch_idx in range(params.PATCHES_PER_VOLUME):
            patch_image3d, patch_mask3d = select_patch(image3d, label3d, params.PATCH_SIZE)

            yield patch_image3d, patch_mask3d


def write_tfrecord_files(mode, max_examples_per_record=512):
    """Create tfrecord files."""
    print('', end='\r')
    if mode == 'training':
        image_data_dir=params.IMAGETR_PATH
        label_data_dir=params.LABELTR_PATH
        tfrecords_dir=params.TRAIN_TFRECORDS_PATH
        num_examples = params.NUM_TRAINING_VOLS*params.PATCHES_PER_VOLUME
    elif mode == 'validation':
        image_data_dir=params.IMAGEVAL_PATH
        label_data_dir=params.LABELVAL_PATH
        tfrecords_dir=params.VALID_TFRECORDS_PATH
        num_examples = params.NUM_VALIDATION_VOLS*params.PATCHES_PER_VOLUME
    elif mode == 'evaluation':
        image_data_dir=params.IMAGEEVAL_PATH
        label_data_dir=params.LABELEVAL_PATH
        tfrecords_dir=params.EVAL_TFRECORDS_PATH
        num_examples = params.NUM_EVALUATION_VOLS*params.PATCHES_PER_VOLUME
    else:
        print('Error: Invalid mode selected.')
    written = 0
    
    num_record_files = int(num_examples / (max_examples_per_record - 1)) + 1
    record_files = [os.path.join(tfrecords_dir, f'{mode}-{i:02}.tfrecord') for i in range(num_record_files)]

    gen = data_generator(image_data_dir, label_data_dir)
    for i, record_file in enumerate(record_files):
        with tf.io.TFRecordWriter(record_file, options='GZIP') as writer:

            for image, mask in gen:

                assert(image.dtype == tf.float32)
                assert(mask.dtype == tf.uint8)

                # images to bytes
                image_bytes = tf.io.serialize_tensor(image)
                mask_bytes = tf.io.serialize_tensor(mask)

                example = serialize_example(image_bytes, mask_bytes, image.shape)
                writer.write(example)

                written += 1
                print(f'mode: {mode}, TFRec file: {i+1:04}/{num_record_files:04}, patches written: {written:05}, img_shp: {image.shape}, mask_shp: {mask.shape}', end='\r')
                # go out of the generator and into another file
                if written % max_examples_per_record == 0:
                    break
                
        
#if __name__ == '__main__':
    #write_tfrecord_files(mode='evaluation', max_examples_per_record=344)
    #write_tfrecord_files(mode='validation', max_examples_per_record=344)
    #write_tfrecord_files(mode='training', max_examples_per_record=344)