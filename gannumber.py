# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:47:53 2017

@author: pegasus
"""
import pylab
import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2

from mnist import MNIST
mndata = MNIST('/home/pegasus/python-mnist/data/')
images_train, labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

seed = 128
rng = np.random.RandomState(seed)
"""
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'Data')

train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
"""
temp=[]

for xx in images_train:
    xx=np.reshape(xx,(28,28))
    temp.append(xx)
train_x = np.stack(temp)

print(train_x.shape)
train_x = train_x / 255.

#img_name = rng.choice(images_train)
#filepath = os.path.join(data_dir, 'Train', 'Images', 'train', img_name)
#print(images_train[0])
"""
x = np.reshape(images_train[0], (28, 28))
#print(x)

plt.imshow(x, interpolation='none', cmap='gray')
plt.show()
"""

g_input_shape = 100 
d_input_shape = (28, 28) 
hidden_1_num_units = 500 
hidden_2_num_units = 500 
g_output_num_units = 784 
d_output_num_units = 1 
epochs = 25 
batch_size = 128

model_1 = Sequential([Dense(units=hidden_1_num_units, input_dim=g_input_shape, activation='relu', kernel_regularizer = L1L2(1e-5, 1e-5)),
                      Dense(units=hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)),
                      Dense(units=g_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5,1e-5)),
                      Reshape(d_input_shape),])

model_2=Sequential([InputLayer(input_shape=d_input_shape), Flatten(), Dense(units=hidden_1_num_units, activation='relu', kernel_regularizer=L1L2(1e-5,1e-5)),
                    Dense(units=hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1e-5,1e-5)),
                    Dense(units=d_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1e-5,1e-5)),])
                    
"""
print(model_1.summary())
print(model_2.summary())
"""

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

gan = simple_gan(model_1, model_2, normal_latent_sampling((100,)))
model = AdversarialModel(base_model=gan,player_params=[model_1.trainable_weights, model_2.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'], loss='binary_crossentropy')
print(gan.summary())
history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=10, batch_size=batch_size)
print(gan.summary())

plt.plot(history.history['player_0_loss'])
plt.plot(history.history['player_1_loss'])
plt.plot(history.history['loss'])