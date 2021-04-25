import os
import glob

class Params():
    
    #File Paths
    DATA_PATH = r'D:\Coding_and_Data\BrainTumor_scans\Task01_BrainTumour\Task01_BrainTumour'
    C_DATA_PATH = r'C:\Users\bnunl\coding_stuff\AIinBioMed'
    WEIGHTS_PATH = os.path.join(DATA_PATH, 'weights2d.h5') # necessary
    HISTORY_PATH = os.path.join(DATA_PATH, 'trainHistoryDict') # necessary
    IMAGETR_PATH = os.path.join(DATA_PATH, 'imagesTr') # necessary
    LABELTR_PATH = os.path.join(DATA_PATH, 'labelsTr') # necessary
    IMAGEVAL_PATH = os.path.join(DATA_PATH, 'imagesVal') # necessary
    LABELVAL_PATH = os.path.join(DATA_PATH, 'labelsVal') # necessary
    IMAGEEVAL_PATH = os.path.join(DATA_PATH, 'imagesEval')
    LABELEVAL_PATH = os.path.join(DATA_PATH, 'labelsEval')
    TRAIN_TFRECORDS_PATH = os.path.join(DATA_PATH, 'Train_TFRecords') # necessary
    VALID_TFRECORDS_PATH = os.path.join(DATA_PATH, 'Valid_TFRecords') # necessary
    EVAL_TFRECORDS_PATH = os.path.join(DATA_PATH, 'Eval_TFRecords')
    #IMAGETR_SLICE_PATH = os.path.join(DATA_PATH, 'imagesSlicesTr')
    #LABELTR_SLICE_PATH = os.path.join(DATA_PATH, 'labelsSlicesTr')
    #IMAGEVAL_SLICE_PATH = os.path.join(DATA_PATH, 'imagesSlicesVal')
    #LABELVAL_SLICE_PATH = os.path.join(DATA_PATH, 'labelsSlicesVal')
    C_TRAIN_TFRECORDS_PATH = os.path.join(C_DATA_PATH, 'Train_TFRecords')
    C_VALID_TFRECORDS_PATH = os.path.join(C_DATA_PATH, 'Valid_TFRecords')
    C_EVAL_TFRECORDS_PATH = os.path.join(C_DATA_PATH, 'Eval_TFRecords')
    

    # Data specifics
    NUM_TRAINING_VOLS = len(os.listdir(IMAGETR_PATH))
    NUM_VALIDATION_VOLS = len(os.listdir(IMAGEVAL_PATH))
    NUM_EVALUATION_VOLS = len(os.listdir(IMAGEEVAL_PATH))
    IMG_SIZE = 224 # for model input (height==width)
    START_SLICE = 18 # in vertical stack
    END_SLICE = 138 # in vertical stack
    CHANNELS_DICT = {'FLAIR': 0, 'T1w': 1, 'T1gd': 2, 'T2w': 3}
    CHANNELS = ['FLAIR', 'T1w', 'T1gd', 'T2w'] #{'FLAIR': 0, 'T1w': 1, 'T1gd': 2, 'T2w': 3}
    NUM_CHANNELS = len(CHANNELS)
    NUM_CLASSES = 4

    # Training specifics
    NUM_EPOCHS = 500
    BATCH_SIZE = 12
    STACK_SIZE = 8
    USE_PRETRAINED_WEIGHTS = True

params = Params()