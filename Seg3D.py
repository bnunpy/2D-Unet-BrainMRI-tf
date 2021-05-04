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

from TFRecord_reader_3D import get_tfrecords_dataset, prepare_dataset
import Unet3D
from configure_3D import params


def train():

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    train_dataset = get_tfrecords_dataset('training')
    valid_dataset = get_tfrecords_dataset('validation')
    train_batches = prepare_dataset(train_dataset, params.BATCH_SIZE, 
                                    buffer_size=200,
                                    repeat=0, 
                                    take=0,
                                    cache=False,
                                    shuffle=True, 
                                    prefetch=False)
    valid_batches = prepare_dataset(valid_dataset, params.BATCH_SIZE, 
                                    buffer_size=200,
                                    repeat=0,
                                    take=0,
                                    cache=False,
                                    shuffle=True, 
                                    prefetch=False)
    
    # call and create model
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = Unet3D.unet3d()
    if(params.USE_PRETRAINED_WEIGHTS):
        model = Unet3D.unet3d(pretrained_weights = params.WEIGHTS_PATH)


    # Compile model
    print('-'*30)
    print('Compiling model...')
    print('-'*30)
    opt = Adam(lr=1E-4)
    weights = np.array([1.,5.,10.,10.])
    weightedloss = Unet3D.weighted_categorical_crossentropy(weights)
    model.compile(loss=weightedloss,
                optimizer=opt,
                metrics=['categorical_crossentropy', 
                        'categorical_accuracy',
                        #'val_categorical_crossentropy',
                        Unet3D.dice_coef_multilabel])


    # callbacks
    checkpoint = ModelCheckpoint(params.WEIGHTS_PATH, 
                                monitor='categorical_accuracy', 
                                verbose=1, 
                                save_best_only=True,
                                save_freq=200,
                                mode='max'
                                )
    earlystopping = EarlyStopping(monitor='categorical_accuracy', 
                                verbose=1, 
                                #min_delta=.005, 
                                patience=10, 
                                mode='max'
                                )
    callbacks_list = [checkpoint, earlystopping]

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(train_batches, 
                        epochs=params.NUM_EPOCHS, 
                        steps_per_epoch=(params.NUM_TRAINING_VOLS*params.PATCHES_PER_VOLUME) // params.BATCH_SIZE,
                        validation_data=valid_batches, 
                        validation_steps=(params.NUM_VALIDATION_VOLS*params.PATCHES_PER_VOLUME) // params.BATCH_SIZE,
                        validation_freq=1,
                        callbacks=callbacks_list,
                        verbose=1)

    with open(params.HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == '__main__':
    train()