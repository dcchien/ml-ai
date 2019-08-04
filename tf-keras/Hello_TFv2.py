# simple example of TF with TFv2 Beta1
# 
import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)
print("Keras version:",keras.__version__)

#need to add this line for TFv2 Beta
tf.compat.v1.disable_eager_execution()

hello = tf.constant('Hello, Tensorflow')

#Initiate a Session & need compat.v1
sess = tf.compat.v1.Session()

print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(70)
print(sess.run(a + b))
