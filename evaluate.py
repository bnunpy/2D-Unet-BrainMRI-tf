import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import Unet2d
import data_loader
import preprocess
from configure import params



# Create a new model instance
model = Unet2d.unet2d(params.WEIGHTS_PATH)

# Restore the weights
#model.load_weights(params.WEIGHTS_PATH)

#test_images, test_labels = data_loader.load_2d_testing_data(data_path)

#test_images, test_labels = preprocess.trim_edges(test_images, test_labels)
eval_generator = data_loader.data_generator(params.IMAGEEVAL_PATH, params.LABELEVAL_PATH)

#x_size, y_size = 256,256
#test_images = preprocess.resize_stack(test_images, x_size, y_size)
#test_labels = preprocess.resize_stack(test_labels, x_size, y_size)

#test_images = test_images.astype('float32')/255
#test_labels = test_labels.astype('float32')

loss, acc = model.evaluate(eval_generator, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

print(model.summary())