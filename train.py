# import mygame
import numpy as np
import pickle
import random
import csv
from nn import neural_net, LossHistory
import os.path
import timeit
import time
import keras
from keras.models import Model
import matplotlib.pyplot as plt
# import pygame
from scipy import misc

# clock = pygame.time.Clock()
batch_size = 2

def load_dataset():
	images = []
	for i in range(6,26):
		images.append( (misc.imread('dataset/'+str(i)+'.png'),i))

	# images = np.asarray(images)
	return images

def train_net(model):
	f = open('data.pickle', 'wb')
	'''
		VISUALIZE


	'''
	desiredLayers = [0]
	desiredOutputs = [model.layers[i].output for i in desiredLayers]
	newModel = Model(model.inputs, desiredOutputs)


	normalise = 32
	train_frames = 10000
	t = 0
	dataset = load_dataset()
	
	while t < train_frames:
		t += 1
		# time.sleep(1)
		batch = random.sample(dataset,batch_size)

		inputs = []
		labels = []

		for i in range(batch_size):
			inputs.append(batch[i][0])
			# inputs.append( process_img_cnn(batch[i][0] ))  
			labels.append(batch[i][1])
			

		inputs = np.asarray(inputs)
		# print(dataset[0])
		# plt.imshow(inputs[0])
		# plt.show()
		# print(inputs,file = f)
		pickle.dump(inputs,f)
		# f.close()
		# np.savetxt('text.txt',inputs)
		labels = np.asarray(labels)
		# print(np.shape(inputs[1]))
		# for layer in model.layers:
		# 	print(layer.get_weights())

		# s

		# plt.savefig('Images/'+str(1)+'.png')
		# state = inputs[0]
		# misc.imshow(state)
		# state = np.expand_dims(state, axis=0)
		# arr = newModel.predict(state)
		# print(np.shape(arr))

		# # print(np.shape(arr))
		# # 	# print('count = ',np.count_nonzero(arr))
		# for filter_ in range(arr.shape[3]):
		# 	# Get the 5x5x1 filter:
		# 	extracted_filter = arr[:, :, :, filter_]
		# 	print(np.shape(extracted_filter))
		# 	# Get rid of the last dimension (hence get 5x5):
		# 	extracted_filter = np.squeeze(extracted_filter)

		# # 	# 	# display the filter (might be very small - you can resize the window)
		# 	misc.imshow(extracted_filter)
		# # 	# 	plt.imshow(arr[0,:,:,1])
		# # 	# 	plt.savefig('Images/'+str(1)+'.png')

		print(np.shape(inputs))
		print("Predicted = ",model.predict( inputs, batch_size=batch_size, verbose=0))
		print("Labels = ",labels)
		model.fit(
			inputs,labels, batch_size=batch_size,
			nb_epoch=1, verbose=1
		)

			
			# plt.savefig('Images/'+str(game_state.num_steps)+'.png')
			
		 # Save the model every 25,000 frames.
		if t % 500== 0:
			model.save_weights('saved-models/' +
							   str(t) + '.h5',
							   overwrite=True)
			print("Saving model %d" % (t))

# def launch_learn():
   
# 	model = neural_net('saved-models/14675')
# 	train_net(model)

def process_img(state):
	return np.reshape(state,(128*128))

def process_img_cnn(state):
	return np.reshape(state,(150,500,3))

if __name__ == "__main__":
	# model = neural_net('saved-models/14675.h5')
	model = neural_net()
	print("loaded?")
	# time.sleep(1)
	train_net(model)
