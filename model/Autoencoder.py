# Autoencoder
#https://gist.github.com/akelleh/d4f4cb6b6ef6dae484d98f6d4eb666db
# fixed bugs - missing TSNE and plt as well asd sns 'size' ==? 'height'
# also move __future__ to the top

from __future__ import print_function
import keras
import keras.datasets.mnist as mnist
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from sklearn.manifold import TSNE
from seaborn import pairplot

batch_size = 128
num_classes = 10
epochs = 3

# the data, split between train and test sets
(x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()

x_train = x_train_original.reshape(60000, 784)
x_test = x_test_original.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_original, num_classes)
y_test = keras.utils.to_categorical(y_test_original, num_classes)

image_in = Input(shape=(784,))
h1 = Dense(64, activation='tanh', input_shape=(784,))(image_in)
d1 = Dropout(0.2)(h1)
h = Dense(32, activation='tanh')(h1)
d2 = Dropout(0.2)(h)
h3 = Dense(64, activation='tanh')(d2)
d3 = Dropout(0.2)(h3)
y_out = Dense(784, activation='tanh')(d3)

model = Model(inputs=image_in, outputs=y_out)
model.summary()

encoder = Model(inputs=image_in, outputs=h)

model.compile(loss='mean_squared_error',
              optimizer=RMSprop())

history = model.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, x_test))
score = model.evaluate(x_test, x_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# predict encoder
print(encoder.predict(x_test)[0])

#
x_pred = encoder.predict(x_test)
reduced_dim = TSNE(n_components=2).fit_transform(x_pred[:2000])

#
df = pd.DataFrame(reduced_dim)
df["target"] = y_test_original[:2000]

# graph
pairplot(x_vars=[0], y_vars=[1], data=df, hue="target", height=5)
plt.show()