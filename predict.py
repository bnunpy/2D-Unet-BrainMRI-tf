import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import tensorflow as tf
import scipy
import h5py
import numpy as np
import nibabel as nib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt

from tfrecord_reader import get_tfrecords_dataset, prepare_dataset
import Unet2d
from configure import params

def predict():
    eval_dataset = get_tfrecords_dataset('evaluation')
    eval_batches = prepare_dataset(eval_dataset, params.BATCH_SIZE, repeat=-1)

    # Create a new model instance
    model = Unet2d.unet2d(params.WEIGHTS_PATH)

    print('-'*30)
    print('Compiling model...')
    print('-'*30)
    opt = Adam(lr=1E-4)
    weights = np.array([1,5,10,10])
    weightedloss = Unet2d.weighted_categorical_crossentropy(weights)
    model.compile(loss=weightedloss,
                optimizer=opt,
                metrics=['categorical_crossentropy', 'categorical_accuracy', Unet2d.dice_coef_multilabel])
    #eval_generator = data_loader.data_generator(params.IMAGEEVAL_PATH, params.LABELEVAL_PATH)

    metrics = model.evaluate(eval_batches, verbose=2)
    #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print(model.summary())
    return metrics

def save_label_sidebyside(labels_truth, labels_pred):
    for i in range(BATCH_SIZE):
        labelt = postprocess_label(labels_truth[i])
        labelp = postprocess_label(labels_pred[i])
        plt.clf()
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(labelt, cmap='gray')
        axarr[0].set_title('Ground Truth')
        axarr[0].axis('off')
        axarr[1].imshow(labelp, cmap='gray')
        axarr[1].set_title('Prediction')
        axarr[1].axis('off')
        f.savefig(os.path.join(PRED_PATH, 'label' + str(i) + '.png'),bbox_inches='tight')


if __name__ == '__main__':
    metrics = predict()