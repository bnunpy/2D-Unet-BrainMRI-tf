import os
import sys
import numpy as np
import random
import nibabel as nib
import tensorflow as tf

import preprocess
from configure import params


def load_2d_training_data(data_path):
    images_file = os.path.join(data_path, 'images_train.npy')
    lables_file = os.path.join(data_path, 'labels_train.npy')
    images_train = np.load(images_file)
    labels_train = np.load(lables_file)
    print('Training images shape: ' + str(images_train.shape))
    print('Training masks shape: ' + str(labels_train.shape))
    return images_train, labels_train


def load_2d_validation_data(data_path):
    images_file = os.path.join(data_path, 'images_train.npy')
    lables_file = os.path.join(data_path, 'labels_train.npy')
    images_validation = np.load(images_file)
    labels_validation = np.load(lables_file)
    print('validation images shape: ' + str(images_validation.shape))
    print('validation masks shape: ' + str(labels_validation.shape))
    return images_validation, labels_validation


def train_data_generator(imgs_path=params.IMAGETR_PATH, lbls_path=params.LABELTR_PATH):
    n = os.listdir(imgs_path)

    img_batch = np.zeros((params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CHANNELS)).astype('float32') #Batch of 2d images
    lbl_batch = np.zeros((params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CLASSES)).astype('float32') #Batch of 2d labels
    
    generating = True
    while generating:

        random_file = np.random.randint(0, len(n))
        image3d = nib.load(os.path.join(imgs_path, n[random_file])) #Size (240,240,1??,4)
        label3d = nib.load(os.path.join(lbls_path, n[random_file])) #Size (240,240,1??)

        for im in range(params.BATCH_SIZE):
            sl_idx = np.random.randint(18, 138)
            image2d = np.asanyarray(image3d.dataobj)[:,:,sl_idx,[params.CHANNELS_DICT[x] for x in params.CHANNELS]]
            label2d = np.asanyarray(label3d.dataobj)[:,:,sl_idx]
            image2d = preprocess.resize_img(image2d)/255.
            label2d = preprocess.resize_lbl(label2d)
            
            img_batch[im] = image2d
            lbl_batch[im] = label2d   

        #img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)     
        #lbl_batch = tf.convert_to_tensor(lbl_batch, dtype=tf.float32) 

        yield img_batch, lbl_batch
        

def valid_data_generator(imgs_path=params.IMAGEVAL_PATH, lbls_path=params.LABELVAL_PATH):
    n = os.listdir(imgs_path)
    
    img_batch = np.zeros((params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CHANNELS)).astype('float32') #Batch of 2d images
    lbl_batch = np.zeros((params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CLASSES)).astype('float32') #Batch of 2d labels
    
    generating = True
    while generating:

        random_file = np.random.randint(0, len(n))
        image3d = nib.load(os.path.join(imgs_path, n[random_file])) #Size (240,240,1??,4)
        label3d = nib.load(os.path.join(lbls_path, n[random_file])) #Size (240,240,1??)

        for im in range(params.BATCH_SIZE):
            sl_idx = np.random.randint(18, 138)
            image2d = np.asanyarray(image3d.dataobj)[:,:,sl_idx,[params.CHANNELS_DICT[x] for x in params.CHANNELS]]
            label2d = np.asanyarray(label3d.dataobj)[:,:,sl_idx]
            image2d = preprocess.resize_img(image2d)/255.
            label2d = preprocess.resize_lbl(label2d)
            
            img_batch[im] = image2d
            lbl_batch[im] = label2d        
        
        #img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)     
        #lbl_batch = tf.convert_to_tensor(lbl_batch, dtype=tf.float32) 
        
        yield img_batch, lbl_batch


def eval_data_generator(imgs_path=params.IMAGEEVAL_PATH, lbls_path=params.LABELEVAL_PATH):
    n = os.listdir(imgs_path)
    
    img_batch = np.zeros((params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CHANNELS)).astype('float32') #Batch of 2d images
    lbl_batch = np.zeros((params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CLASSES)).astype('float32') #Batch of 2d labels
    
    generating = True
    while generating:

        random_file = np.random.randint(0, len(n))
        image3d = nib.load(os.path.join(imgs_path, n[random_file])) #Size (240,240,1??,4)
        label3d = nib.load(os.path.join(lbls_path, n[random_file])) #Size (240,240,1??)

        for im in range(params.BATCH_SIZE):
            sl_idx = np.random.randint(18, 138)
            image2d = np.asanyarray(image3d.dataobj)[:,:,sl_idx,[params.CHANNELS_DICT[x] for x in params.CHANNELS]]
            label2d = np.asanyarray(label3d.dataobj)[:,:,sl_idx]
            image2d = preprocess.resize_img(image2d)/255.
            label2d = preprocess.resize_lbl(label2d)
            
            img_batch[im] = image2d
            lbl_batch[im] = label2d        
        
        #img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)     
        #lbl_batch = tf.convert_to_tensor(lbl_batch, dtype=tf.float32) 
        
        yield img_batch, lbl_batch

'''
def write_TFRecord():
    pass

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

def serialize_example(image, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(image, label):
    tf_string = tf.py_function(
        serialize_example,
        (image, label),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar


#dataset = tf.data.Dataset.from_tensor_slices((image, label))

#filename = 'D:\Coding_and_Data\BrainTumor_scans\Task01_BrainTumour\Task01_BrainTumour\imagesTr\TRAINING.tfrecord'
img_files = os.listdir(params.IMAGETR_PATH)
lbl_files = os.listdir(params.LABELTR_PATH)
#generating = True

for i, (img, lbl) in enumerate(zip(img_files, lbl_files)):
    filename = f'D:\Coding_and_Data\BrainTumor_scans\Task01_BrainTumour\Task01_BrainTumour\imagesTr\TRAINING{i}.tfrecord'
    
    image3d = nib.load(os.path.join(params.IMAGETR_PATH, img)) #Size (240,240,1??,4)
    image3d = np.asanyarray(image3d.dataobj)
    label3d = nib.load(os.path.join(params.LABELTR_PATH, lbl)) #Size (240,240,1??)
    label3d = np.asanyarray(label3d.dataobj)

    features_dataset = tf.data.Dataset.from_tensor_slices((image3d, label3d))
    serialized_features_dataset = features_dataset.map(tf_serialize_example)

    def generator():
        for features in features_dataset:
            yield serialize_example(*features)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())
    
    with tf.io.TFRecordWriter(filename) as writer:
        #writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(serialized_features_dataset)

    #example = serialize_example(image3d, label3d)
    #writer.write(example)
'''

        




#img_batch, lbl_batch = data_generator(params.IMAGETR_PATH, params.LABELTR_PATH)
#print(img_batch.shape, lbl_batch.shape)
#print(sys.getsizeof(data_generator(params.IMAGETR_PATH, params.LABELTR_PATH)))

#dataset = tf.data.Dataset.list_files(params.IMAGETR_PATH + "\\*.gz")
#for element in dataset.as_numpy_iterator():
#  print(element)
#print(dataset.batch(2))