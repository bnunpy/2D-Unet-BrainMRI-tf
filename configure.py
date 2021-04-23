import os

class Params():
    
    #File Paths
    DATA_PATH = r'D:\Coding_and_Data\BrainTumor_scans\Task01_BrainTumour\Task01_BrainTumour'
    WEIGHTS_PATH = os.path.join(DATA_PATH, 'weights2d.h5')
    HISTORY_PATH = os.path.join(DATA_PATH, 'trainHistoryDict')
    IMAGETR_PATH = os.path.join(DATA_PATH, 'imagesTr')
    LABELTR_PATH = os.path.join(DATA_PATH, 'labelsTr')
    IMAGEVAL_PATH = os.path.join(DATA_PATH, 'imagesVal')
    LABELVAL_PATH = os.path.join(DATA_PATH, 'labelsVal')
    IMAGEEVAL_PATH = os.path.join(DATA_PATH, 'imagesEval')
    LABELEVAL_PATH = os.path.join(DATA_PATH, 'labelsEval')
    TRAIN_TFRECORDS_PATH = os.path.join(DATA_PATH, 'Train_TFRecords')
    VALID_TFRECORDS_PATH = os.path.join(DATA_PATH, 'Valid_TFRecords')

    # Data specifics
    NUM_TRAINING_IMGS = 350
    NUM_VALIDATION_IMGS = 130
    IMG_SIZE = 256 # for model input
    CHANNELS_DICT = {'FLAIR': 0, 'T1w': 1, 'T1gd': 2, 'T2w': 3}
    CHANNELS = ['FLAIR', 'T1w', 'T1gd', 'T2w'] #{'FLAIR': 0, 'T1w': 1, 'T1gd': 2, 'T2w': 3}
    NUM_CHANNELS = len(CHANNELS)
    NUM_CLASSES = 4

    # Training specifics
    NUM_EPOCHS = 20
    BATCH_SIZE = 8
    STACK_SIZE = 8
    USE_PRETRAINED_WEIGHTS = False

params = Params()