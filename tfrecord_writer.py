import os
import sys
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import glob
import nibabel as nib

from configure import params

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
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'num_channels': _int64_feature(image_shape[2])
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def data_generator(imgs_path, lbls_path):
    
    filenames = os.listdir(imgs_path)

    for filename in filenames:
        image_file = os.path.join(imgs_path, filename)
        label_file = os.path.join(lbls_path, filename)
        image3d = nib.load(image_file)        
        label3d = nib.load(label_file)
        for sl_idx in range(params.START_SLICE, params.END_SLICE):
            image2d = np.asanyarray(image3d.dataobj, dtype=np.float32)[8:232,8:232,sl_idx,:]
            label2d = np.asanyarray(label3d.dataobj, dtype=np.uint8)[8:232,8:232,sl_idx]
            image2d = tf.convert_to_tensor(image2d, dtype=tf.float32)
            label2d = tf.convert_to_tensor(label2d, dtype=tf.uint8)

            yield image2d, label2d


def write_tfrecord_files(mode, max_examples_per_record=1000):
    """Create tfrecord files."""
    print('', end='\r')
    if mode == 'training':
        image_data_dir=params.IMAGETR_PATH
        label_data_dir=params.LABELTR_PATH
        tfrecords_dir=params.TRAIN_TFRECORDS_PATH
        num_examples = params.NUM_TRAINING_VOLS*(params.END_SLICE-params.START_SLICE)
    elif mode == 'validation':
        image_data_dir=params.IMAGEVAL_PATH
        label_data_dir=params.LABELVAL_PATH
        tfrecords_dir=params.VALID_TFRECORDS_PATH
        num_examples = params.NUM_VALIDATION_VOLS*(params.END_SLICE-params.START_SLICE)
    elif mode == 'evaluation':
        image_data_dir=params.IMAGEEVAL_PATH
        label_data_dir=params.LABELEVAL_PATH
        tfrecords_dir=params.EVAL_TFRECORDS_PATH
        num_examples = params.NUM_EVALUATION_VOLS*(params.END_SLICE-params.START_SLICE)
    else:
        print('Error: Invalid mode selected.')
    written = 0
    
    num_record_files = int(num_examples / (max_examples_per_record - 1)) + 1
    record_files = [os.path.join(tfrecords_dir, f'{mode}-{i:02}.tfrecord') for i in range(num_record_files)]

    gen = data_generator(image_data_dir, label_data_dir)
    for i, record_file in enumerate(record_files):
        #tq.set_postfix(record_file=record_file)
        with tf.io.TFRecordWriter(record_file, options='GZIP') as writer:

            for image, mask in gen:

                maskt = tf.convert_to_tensor(mask)
                imaget = tf.convert_to_tensor(image)

                assert(imaget.dtype == tf.float32)
                assert(maskt.dtype == tf.uint8)

                # images to bytes
                image_bytes = tf.io.serialize_tensor(imaget)
                mask_bytes = tf.io.serialize_tensor(maskt)

                example = serialize_example(image_bytes, mask_bytes, image.shape)
                writer.write(example)

                written += 1
                print(f'mode: {mode}, TFRec file: {i+1:03}/{num_record_files:03}, slices written: {written:05}, img_shp: {image.shape}, mask_shp: {mask.shape}', end='\r')
                # go out of the generator and into another file
                if written % max_examples_per_record == 0:
                    break
                
        
if __name__ == '__main__':
    write_tfrecord_files(mode='evaluation', max_examples_per_record=480)
    #write_tfrecord_files(mode='validation', max_examples_per_record=480)
    #write_tfrecord_files(mode='training', max_examples_per_record=480)
    



'''    #write tfrecords
    file_num = 0
    tfrec_num = 1
    for filename in glob.iglob(f'{image_data_dir}\*'):
        subject_name = filename[-16:]
        file_num += 1
        
        #print('----------------------------------------------------------------------------',filename)
        image_file = os.path.join(image_data_dir, filename)
        label_file = os.path.join(label_data_dir, filename)
        
        image2d = np.load(image_file)        
        label2d = np.load(label_file)

        tfrecord_filename = os.path.join(tfrecords_dir, (f'{mode}-{tfrec_num}' + '.tfrecords'))
        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            tf_example = image_example(image2d, label2d, int(filename.split('.')[0][-1]))
            writer.write(tf_example.SerializeToString())
            print(f'tfrec: {tfrec_num}, file_num: {file_num:04}', end='\r')


        if file_num > 1000: 
            #print(f'file: {subject_name}, TFRecord file: {tfrecord_filename}', end='\r')
            tfrec_num += 1
            file_num = 0    '''