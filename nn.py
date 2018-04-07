"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD,RMSprop
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback
import numpy as np
import random
from collections import deque
import keras 
from keras.models import Sequential, load_model,Model
from keras import optimizers
from keras.layers import Dense,Dropout,Activation,Concatenate,Merge
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras import initializers
from keras.layers import Conv2D,MaxPooling2D,Flatten,GlobalAveragePooling2D
import matplotlib.pyplot as plt
from keras.utils import plot_model

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


def neural_net( load=''):
	model = Sequential()
	num_classes = 1
	inp_shape = (500,150,3)
	# inp_shape = (128*128)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
					 activation='relu',
					 input_shape=inp_shape))
# # # 
	model.add(Conv2D(32, (5, 5), activation='relu'))
# # 
	# model.add(Conv2D(64, (5, 5), activation='relu'))

	# model.add(Conv2D(64, (5, 5), activation='relu'))



	# # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# # model.add(Conv2D(64, (5, 5), activation='relu'))
	# # model.add(MaxPooling2D(pool_size=(2, 2)))
	
	# model.add(Conv2D(128, (5, 5), activation='relu'))
	# model.add(Conv2D(128, (5, 5), activation='relu'))
	# model.add(Conv2D(128, (5, 5), activation='relu'))
	# model.add(Conv2D(128, (5, 5), activation='relu'))
	model.add(Flatten())
	# model.add(GlobalAveragePooling2D())
	# model.add(Dense(1000, init='lecun_uniform',activation='relu'))
	# model.add(Dense(5,activation='relu',input_dim=inp_shape))
	# model.add(Dense(1000,init='lecun_uniform',activation='relu'))
	model.add(Dense(64,activation='relu'))
	# model.add(Dense(256, init='lecun_uniform',activation='relu'))

	model.add(Dense(64, activation='relu'))
	# model.add(Dense(256, init='lecun_uniform',activation='relu'))
	model.add(Dense(num_classes, activation='linear'))
  




	rms = RMSprop(lr=0.0001)
	model.compile(loss='mse', optimizer=rms)

	# plot_model(model, to_file='model.png',show_shapes = True)



	if load:
		model.load_weights(load)
		print("load")

	return model
