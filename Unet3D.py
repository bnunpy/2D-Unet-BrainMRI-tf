from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Dropout, UpSampling3D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K

from configure_3D import params


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels=params.NUM_CLASSES):
    dice=0.
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels

def dice_multilabel_loss(y_true, y_pred):
    loss = 1 - dice_coef_multilabel(y_true, y_pred)
    return loss

def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def conv3D_DoubleLayer(input, nfilters):
    conv1 = Conv3D(nfilters, 3, padding = 'same', kernel_initializer = 'he_normal')(input)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    conv2 = Conv3D(nfilters, 3, padding = 'same', kernel_initializer = 'he_normal')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    return act2

# data_format='channels_last',
def unet3d(pretrained_weights=None):
    inputs = Input((params.PATCH_SIZE[0], params.PATCH_SIZE[1], params.PATCH_SIZE[2], params.NUM_CHANNELS))
    conv1 = conv3D_DoubleLayer(inputs, 32)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv3D_DoubleLayer(pool1, 64)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv3D_DoubleLayer(pool2, 128)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv3D_DoubleLayer(pool3, 256)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = conv3D_DoubleLayer(pool4, 512)

    up6 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = 2)(conv5))
    merge6 = concatenate([conv4,up6], axis = 4)
    conv6 = conv3D_DoubleLayer(merge6, 256)

    up7 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = 2)(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = conv3D_DoubleLayer(merge7, 128)

    up8 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = 2)(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = conv3D_DoubleLayer(merge8, 64)

    up9 = Conv3D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = 2)(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = conv3D_DoubleLayer(merge9, 32)

    conv10 = Conv3D(4, 1, activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
