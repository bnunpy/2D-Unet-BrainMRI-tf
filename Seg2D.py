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

from tfrecord_reader import get_tfrecords_dataset, prepare_dataset
#import data_loader
#import preprocess
import Unet2d
from configure import params


def train():

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    train_dataset = get_tfrecords_dataset('training')
    valid_dataset = get_tfrecords_dataset('validation')
    train_batches = prepare_dataset(train_dataset, params.BATCH_SIZE)
    valid_batches = prepare_dataset(valid_dataset, params.BATCH_SIZE)
    
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
    weights = np.array([1.,5.,8.,8.])
    weightedloss = Unet2d.weighted_categorical_crossentropy(weights)
    model.compile(loss=weightedloss,
              optimizer=opt,
              metrics=['categorical_crossentropy', 'categorical_accuracy', Unet2d.dice_coef_multilabel])


    # callbacks
    checkpoint = ModelCheckpoint(params.WEIGHTS_PATH, monitor='loss', verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='loss', verbose=1, min_delta=.001, patience=10, mode='min')
    callbacks_list = [checkpoint, earlystopping]


    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(train_batches, epochs=params.NUM_EPOCHS, steps_per_epoch=(16*params.NUM_TRAINING_VOLS//params.BATCH_SIZE),
              validation_data=valid_batches, validation_steps=(16*params.NUM_VALIDATION_VOLS//params.BATCH_SIZE),
              callbacks=callbacks_list,
              verbose=1)

    with open(params.HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    train()