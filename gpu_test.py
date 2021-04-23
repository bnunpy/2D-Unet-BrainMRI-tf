from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np
import os
#from keras.preprocessing.image import ImageDataGenerator

import data_loader
import preprocess
import Unet2d
from configure import params

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if not tf.test.gpu_device_name(): 
    warnings.warn('No GPU found')
else: 
    print('Default GPU device: {}' .format(tf.test.gpu_device_name()))
