''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
b_fashion.py

Run:
floyd run \
--data jengtallis/datasets/fashion/1:/data \
"python b_fashion.py"

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import keras
import gzip
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json
from keras import backend as be
from sklearn.model_selection import train_test_split


'''
########################################################
# ============= FASHION_MNIST LOADER  ===============
'''
def load_mnist(path, kind='train'):
	labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)

	images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

	return images, labels


'''
########################################################
# ============= Learning Rate Scheduler  ===============
'''
def scheduler(epoch):
	learning_rate_init = 0.001
	if epoch > 55:
		learning_rate_init = 0.0002
	if epoch > 70:
		learning_rate_init = 0.00005
	return learning_rate_init

'''
########################################################
# ================== Data Directory  ===================
''' 
pathjoin = os.path.join

# Data Directory
DATA_DIR = '/data'

# Output File Path
LOG_DIR = '/output/tb_log/'
WEIGHT_DIR = '/output/b_weights/'
model_dir = '/output/b_models/'
train_id = '1'
model_name = 'model_b_fashion_' + train_id + '.json'
weight_name = 'weights_b_fashion_' + train_id + '.h5'
MODEL_FILE = pathjoin(model_dir, model_name)
WEIGHT_FILE = pathjoin(WEIGHT_DIR, weight_name)


'''
#########################################################
# ================= Model Parameters  ===================
'''
batch_size	= 128
epochs		= 20 #60
test_size	= 0.3

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
def trainer(batch_size, epochs, test_size, DATA_DIR, LOG_DIR, MODEL_FILE, WEIGHT_FILE):

	# ==================== Data  ====================

	# ==================== data definition =====================
	size = 28
	channel = 3
	input_shape = (size, size, channel)

	# === fine-grained classes ===
	n_fg = 10

	# Training data: Input (X) and Expected Output (Y)
	X_train, Y_train = load_mnist(DATA_DIR, kind='train')
	X_train = X_train.astype('float32')
	X_train = (X_train-np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
	X_train = X_train.reshape(-1,size,size)
	X_train = np.stack((X_train,)*channel, -1)

	# Test data
	X_test, Y_test = load_mnist(DATA_DIR, kind='t10k')
	X_test = X_test.astype('float32')
	X_test = (X_test-np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
	X_test = X_test.reshape(-1,size,size)
	X_test = np.stack((X_test,)*channel, -1)

	# Make classes one hot
	Y_train = keras.utils.to_categorical(Y_train, n_fg)
	Y_test = keras.utils.to_categorical(Y_test, n_fg)


	# ==================== Create B-CNN Model  ====================
	batch_size = batch_size
	epochs = epochs
	print("Building B-CNN Model")

	input_imgs = Input(shape=input_shape, name='input')

	# ==================== Block 1  ====================
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_imgs)
	x = BatchNormalization()(x)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# ==================== Block 2  ====================
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# ==================== Block 3  ====================
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# ==================== Block 4  ====================
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# ==================== Fine-Grained Block  ====================
	x = Flatten(name='flatten')(x)
	x = Dense(1024, activation='relu', name='fc_fashion_1')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu', name='fc2')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	fine_pred = Dense(n_fg, activation='softmax', name='pred_fashion')(x)

	model = Model(input_imgs, fine_pred, name='b_fashion')

	# ==================== Compile Model  ====================
	sgd = optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)
	model.compile(	loss='categorical_crossentropy', 
					optimizer=sgd,
					# optimizer=keras.optimizers.Adadelta(),
					metrics=['accuracy'])

	tb_cb = TensorBoard(LOG_DIR=LOG_DIR, histogram_freq=0)
	change_lr = LearningRateScheduler(scheduler)
	cbks = [change_lr,tb_cb]
	
	model.summary()

	# ==================== Train CNN Model  ====================

	# Fit the model
	print("Start Training")

	train = model.fit(	X_train, Y_train,
				batch_size=batch_size,
				epochs=epochs,
				verbose=1,
				callbacks=cbks,
				validation_data=(X_test, Y_test))

	print("Finish Training")

	# evaluate the model
	print("Evaluation Result:")
	scores = model.evaluate(X_test, Y_test, verbose=0)
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

	jsonfile = MODEL_FILE
	hdf5file = WEIGHT_FILE

	# serialize model to JSON
	model_json = model.to_json()
	with open(jsonfile, "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights(hdf5file)
	print("b_fashion Model saved")
	print('===============================\n')


########## trainer trains model #########
trainer(batch_size, epochs, test_size, DATA_DIR, LOG_DIR, MODEL_FILE, WEIGHT_FILE)
