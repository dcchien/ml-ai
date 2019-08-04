# simple example of TF with TFv2 Beta1
# 
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)
print("Keras version:",keras.__version__)

# @tf.function decorator
@tf.function
def add(a, b):
  return a + b

print(add(tf.constant(20), tf.constant(80)))
print(add(10,60))

print(add(tf.zeros([2, 2]), tf.zeros([2, 2])))
print(add(tf.ones([2, 2]), tf.ones([2, 2])))

#need to add this line for TFv2 Beta if you use compat.v1
#disable eagle execution
tf.compat.v1.disable_eager_execution()

hello = tf.constant('Hello, Tensorflow')

#Initiate a Session & need compat.v1
sess = tf.compat.v1.Session()

print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(70)
print("tfv1 session run", sess.run(a + b))
