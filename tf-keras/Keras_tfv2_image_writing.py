# Python 3.7, TFv2.0B and Keras 2.2.4
# MNIST Demo using Keras CNN
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# https://pages.databricks.com/rs/094-YMS-629/images/Keras%20MNIST%20CNN.html
# Use TensorFlow Backend
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(40) # For reproducibility

# Check for any available GPUs ? 
# got error msg: module 'tensorflow._api.v1.keras.backend' has no attribute 'tensorflow_backend' 
# K.tensorflow_backend._get_available_gpus()
# -----------------------------------------------------------
# Hyperparameters
batch_size = 128
num_classes = 10
epochs = 12

# -----------------------------------------------------------
# Image Datasets

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#('x_train shape:', (60000, 28, 28, 1))
#(60000, 'train samples')
#(10000, 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# One-Hot Vector for y_train = 25168 representing the number 9 
# The nth-digit will be represented as a vector which is 1 in the nth dimensions. 
y_train[25168,:]


# This is the extracted array for x_train = 25168 from the training matrix
xt_25168 = x_train[25168,:]

#print(xt_25168)

def runCNN(activation, verbose):
  # Building up our CNN
  model = Sequential()
  
  # Convolution Layer
  model.add(Conv2D(32, kernel_size=(3, 3),
                 activation=activation,
                 input_shape=input_shape)) 
  
  # Convolution layer
  model.add(Conv2D(64, (3, 3), activation=activation))
  
  # Pooling with stride (2, 2)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  # Delete neuron randomly while training (remain 75%)
  #   Regularization technique to avoid overfitting
  model.add(Dropout(0.25))
  
  # Flatten layer 
  model.add(Flatten())
  
  # Fully connected Layer
  model.add(Dense(128, activation=activation))
  
  # Delete neuron randomly while training (remain 50%) 
  #   Regularization technique to avoid overfitting
  model.add(Dropout(0.5))
  
  # Apply Softmax
  model.add(Dense(num_classes, activation='softmax'))

  # Loss function (crossentropy) and Optimizer (Adadelta)
  model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

  # Fit our model
  model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=verbose,
          validation_data=(x_test, y_test))

  # Evaluate our model
  score = model.evaluate(x_test, y_test, verbose=0)
  
  # Return
  return score

# Take a long time to run it on Laptop....using GPU
#    
score_relu = runCNN('relu', 1)
print('Test loss:', score_relu[0])
print('Test accuracy:', score_relu[1])

score_sigmoid = runCNN('sigmoid', 0)
print('Test loss:', score_sigmoid[0])
print('Test accuracy:', score_sigmoid[1])
