import os
import sys
import tensorflow as tf
import numpy as np
import SimpleITK as sitk

from configure import params


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

   

def generate_files(mode):
    """Create tfrecord files."""
    if mode == 'training':
        image_data_dir=params.IMAGETR_PATH, 
        label_data_dir=params.LABELTR_PATH,
        tfrecords_dir=params.TRAIN_TFRECORDS_PATH
    elif mode == 'validation':
        image_data_dir=params.IMAGEVAL_PATH, 
        label_data_dir=params.LABELVAL_PATH,
        tfrecords_dir=params.VALID_TFRECORDS_PATH
    else:
        raise('Error: Invalid mode selected.')

    #write tfrecords
    file_num = 0
    tfrec_num = 1
    for filename in os.listdir(image_data_dir):
        subject_name = filename.split('.')[0]    
        file_num += 1

        image_file = os.path.join(image_data_dir, filename)
        label_file = os.path.join(label_data_dir, filename)
        
        ft = sitk.ReadImage(image_file)
        image = sitk.GetArrayFromImage(ft)

        fl = sitk.ReadImage(label_file)
        label = sitk.GetArrayFromImage(fl)

        if file_num > 8:
            tfrec_num += 1
            file_num = 0
        tfrecord_filename = os.path.join(tfrecords_dir, (f'{filename}' + '.tfrecords'))
        print(f'file: {filename}, slice: {sl:03}', end='\r')


        img_raw = image[:,:,18:138,:].tobytes()
        lbl_raw = label[:,:,18:138].tobytes()


        feature_list = []
        feature = {}
        feature['images'] = _bytes_feature(tf.io.serialize_tensor(img_raw))
        feature['labels'] = _bytes_feature(tf.io.serialize_tensor(lbl_raw))
        feature['channel'] = _int64_feature(value)

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            writer.write(example.SerializeToString())
        
        
if __name__ == '__main__':
    generate_files(mode='training')