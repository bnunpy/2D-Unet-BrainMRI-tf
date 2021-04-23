import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import scipy
import h5py
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History

import data_loader
import preprocess
import Unet2d
from configure import params


def train():

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    #train_generator = data_loader.data_generator(params.IMAGETR_PATH, params.LABELTR_PATH) # does the resizing
    #validation_generator = data_loader.data_generator(params.IMAGEVAL_PATH, params.LABELVAL_PATH) # does the resizing
    train_gen_dataset = tf.data.Dataset.from_generator(data_loader.train_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CHANNELS), 
                            dtype=tf.float32), # image batch
            tf.TensorSpec(shape=(params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CLASSES), 
                            dtype=tf.float32))) # label batch
    valid_gen_dataset = tf.data.Dataset.from_generator(data_loader.valid_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CHANNELS), 
                            dtype=tf.float32), # image batch
            tf.TensorSpec(shape=(params.BATCH_SIZE, params.IMG_SIZE, params.IMG_SIZE, params.NUM_CLASSES), 
                            dtype=tf.float32))) # label batch

    # call and create model
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = Unet2d.unet2d()
    if(params.USE_PRETRAINED_WEIGHTS):
        model = Unet2d.unet2d(pretrained_weights = params.WEIGHTS_PATH)


    # Compile model
    print('-'*30)
    print('Compiling model...')
    print('-'*30)
    opt = Adam(lr=1E-4)
    weights = np.array([1,5,10,10])
    weightedloss = Unet2d.weighted_categorical_crossentropy(weights)
    model.compile(loss=weightedloss,
              optimizer=opt,
              metrics=['categorical_crossentropy', 'categorical_accuracy', Unet2d.dice_coef_multilabel])


    # callbacks
    checkpoint = ModelCheckpoint(params.WEIGHTS_PATH, monitor='val_categorical_crossentropy', verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_categorical_crossentropy', verbose=1, min_delta=0.005, patience=5, mode='min')
    callbacks_list = [checkpoint, earlystopping]


    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(train_gen_dataset, epochs=params.NUM_EPOCHS, steps_per_epoch=(params.NUM_TRAINING_IMGS//params.BATCH_SIZE),
              validation_data=valid_gen_dataset, validation_steps=(params.NUM_VALIDATION_IMGS//params.BATCH_SIZE),
              callbacks=callbacks_list,
              verbose=1)

    with open(params.HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    train_gen_dataset.prefetch(1)
    valid_gen_dataset.prefetch(1)

if __name__ == '__main__':
    train()
