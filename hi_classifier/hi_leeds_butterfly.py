
''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
hi_leeds_butterfly.py

Usage: 
Under 	parent of hi_fgic
Run 	python3 -m fgic.hi_classifier.hi_leeds_butterfly.py

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

# import libraries
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

# import package functions
from ..img_util.segment import segment


'''
########################################################
# ==================== Data Files  ====================
'''



'''
#########################################################
# ==================== Prepare Data  ====================
''' 


'''
#################################################################
# ==================== Load Model from Disk  ====================
'''
def load_model(jsonfile, hdf5file, X, Y):

	# ==================== Load the Model ====================

	# load json and create model
	json_file = open(jsonfile, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	print("Model created")

	# load weights into new model
	model.load_weights(hdf5file)
	print("Model weights loaded from disk")


	# ==================== Run the Model ====================

	# compile loaded model
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# evaluate the model
	print("Evaluation Result:")
	scores = model.evaluate(X, Y, verbose=0)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))
	p = model.predict(X[0])
	print(p)


'''
########################################################
# ==================== CNN Trainer  ====================
'''
def trainer(batch_size, epochs):

	# ==================== Data  ====================


	# Training data: Input (X) and Expected Output (Y)
	train_X, valid_X, test_X, train_Y, valid_Y, test_Y = ???

	train_X = train_X.reshape(-1, sensor_h, sensor_w, 1)
	valid_X = valid_X.reshape(-1, sensor_h, sensor_w, 1)
	test_X = test_X.reshape(-1, sensor_h, sensor_w, 1)

	train_X = train_X.astype('float32')
	valid_X = valid_X.astype('float32')
	test_X = test_X.astype('float32')
	
	train_X = train_X / 255
	valid_X = valid_X / 255
	test_X = test_X / 255

	# ==================== Create CNN Model  ====================
	batch_size = batch_size
	epochs = epochs
	print("Building B-CNN Model")

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(sensor_h,sensor_w,1),padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D((2, 2),padding='same'))
	model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
	model.add(LeakyReLU(alpha=0.1))                
	model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	model.add(Flatten())
	model.add(Dense(128, activation='linear'))
	model.add(LeakyReLU(alpha=0.1))                  
	model.add(Dense(case_size, activation='softmax'))
	model.add(Dense(case_size, activation='sigmoid'))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	# ==================== Train CNN Model  ====================

	# Fit the model
	print("Start Training")
	train = model.fit(train_X, train_Y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_Y))
	print("Finish Training")

	# evaluate the model
	print("Evaluation Result:")
	scores = model.evaluate(test_X, test_Y, verbose=0)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

	# ==================== Plot result  ====================
	accuracy = train.history['acc']
	val_accuracy = train.history['val_acc']
	loss = train.history['loss']
	val_loss = train.history['val_loss']
	epochs = range(len(accuracy))
	plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
	plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

	# ==================== Save CNN Model  ====================

	jsonfile = 'aavia_algo/trained_model/pc.json'
	hdf5file = 'aavia_algo/trained_model/pc.h5'

	# serialize model to JSON
	model_json = model.to_json()
	with open(jsonfile, "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights(hdf5file)
	print("Hi_leeds_butterfly Model saved")
	print('===============================\n')


########## fix random seed for reproducibility #########
np.random.seed(26)

########## trainer trains model #########
trainer(batch_size, epochs, case_h, case_w, sensor_h, sensor_w, files, bs, cs)
