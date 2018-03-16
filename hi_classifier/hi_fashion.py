''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
hi_fashion.py

Usage: 
Under 	parent of hi_fgic
Run 	python3 -m hi_fgic.hi_classifier.hi_fashion.py

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
# ============= Loss Weight Modifier ===============
'''
class LossWeightsModifier(keras.callbacks.Callback):
  def __init__(self, alpha, beta, gamma):
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
  def on_epoch_end(self, epoch, logs={}):
    if epoch == 8:
      be.set_value(self.alpha, 0.1)
      be.set_value(self.beta, 0.8)
      be.set_value(self.gamma, 0.1)
    if epoch == 18:
      be.set_value(self.alpha, 0.1)
      be.set_value(self.beta, 0.2)
      be.set_value(self.gamma, 0.7)
    if epoch == 28:
      be.set_value(self.alpha, 0)
      be.set_value(self.beta, 0)
      be.set_value(self.gamma, 1)

'''
########################################################
# ================== Data Directory  ===================
''' 
# Set relative path to absolute
here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
pathjoin = os.path.join

# Data Directory
data_dir = '../data/fashion'
DATA_DIR = here(data_dir)

# Output File Path
log_dir = '../tb_log/'
weight_dir = '../hi_weights/'
model_dir = '../hi_models/'
train_id = '2'
model_name = 'model_hi_fashion_' + train_id + '.json'
weight_name = 'weights_hi_fashion_' + train_id + '.h5'
model_file = os.path.join(model_dir, model_name)
weight_file = os.path.join(weight_dir, weight_name)

LOG_DIR = here(log_dir)
WEIGHT_fILE = here(weight_file)
MODEL_FILE = here(model_file)


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
def trainer(batch_size, epochs, test_size, DATA_DIR, LOG_DIR, MODEL_FILE, WEIGHT_fILE):

	# ==================== Data  ====================

	# ==================== data definition =====================
	size = 28
	channel = 3
	input_shape = (size, size, channel)

	# === coarse 1 classes ===
	n_c1 = 2
	# === coarse 2 classes ===
	n_c2 = 6
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


	# ============== coarse 2 labels ===============
	fg_parent = {
		0:1,
		1:2, 
		2:0,
		3:3,
		4:0,
		5:5,
		6:1,
		7:5,
		8:4, 
		9:5
	}

	Y_c2_train = np.zeros((Y_train.shape[0], n_c2)).astype("float32")
	for i in range(Y_c2_train.shape[0]):
		Y_c2_train[i][fg_parent[np.argmax(Y_train[i])]] = 1.0

	Y_c2_test = np.zeros((Y_test.shape[0], n_c2)).astype("float32")
	for i in range(Y_c2_test.shape[0]):
		Y_c2_test[i][fg_parent[np.argmax(Y_test[i])]] = 1.0

	# ============== coarse 1 labels ===============
	c2_parent = {
		0:0, 1:0, 2:0, 3:0, 
		4:1, 5:1
	}

	Y_c1_train = np.zeros((Y_c2_train.shape[0], n_c1)).astype("float32")
	for i in range(Y_c1_train.shape[0]):
		Y_c1_train[i][c2_parent[np.argmax(Y_c2_train[i])]] = 1.0

	Y_c1_test = np.zeros((Y_c2_test.shape[0], n_c1)).astype("float32")
	for i in range(Y_c1_test.shape[0]):
		Y_c1_test[i][c2_parent[np.argmax(Y_c2_test[i])]] = 1.0


	# ==================== Create B-CNN Model  ====================
	batch_size = batch_size
	epochs = epochs
	print("Building B-CNN Model")

	input_imgs = Input(shape=input_shape, name='input')

	alpha = be.variable(value=0.98, dtype="float32", name="alpha") # A1 in paper
	beta  = be.variable(value=0.01, dtype="float32", name="beta") # A2 in paper
	gamma = be.variable(value=0.01, dtype="float32", name="gamma") # A3 in paper

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

	# ==================== Coarse 1 Branch  ====================
	c1_bch = Flatten(name='c1_flatten')(x)
	c1_bch = Dense(256, activation='relu', name='c1_fc_fashion_1')(c1_bch)
	c1_bch = BatchNormalization()(c1_bch)
	c1_bch = Dropout(0.5)(c1_bch)
	c1_bch = Dense(256, activation='relu', name='c1_fc2')(c1_bch)
	c1_bch = BatchNormalization()(c1_bch)
	c1_bch = Dropout(0.5)(c1_bch)
	c1_pred = Dense(n_c1, activation='softmax', name='c1_pred_fashion')(c1_bch)

	# ==================== Block 3  ====================
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# ==================== Coarse 2 Branch  ====================
	c2_bch = Flatten(name='c2_flatten')(x)
	c2_bch = Dense(512, activation='relu', name='c2_fc_fashion_1')(c2_bch)
	c2_bch = BatchNormalization()(c2_bch)
	c2_bch = Dropout(0.5)(c2_bch)
	c2_bch = Dense(512, activation='relu', name='c2_fc2')(c2_bch)
	c2_bch = BatchNormalization()(c2_bch)
	c2_bch = Dropout(0.5)(c2_bch)
	c2_pred = Dense(n_c2, activation='softmax', name='c2_pred_fashion')(c2_bch)

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
	fg_pred = Dense(n_fg, activation='softmax', name='pred_fashion')(x)
	
	model = Model(input=input_imgs, output=[c1_pred, c2_pred, fg_pred], name='hi_fashion')

	# ==================== Compile Model  ====================
	sgd = optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)
	model.compile(	loss='categorical_crossentropy', 
					optimizer=sgd, 
					loss_weights=[alpha, beta, gamma],
					# optimizer=keras.optimizers.Adadelta(),
					metrics=['accuracy'])

	tb_cb = TensorBoard(log_dir=LOG_DIR, histogram_freq=0)
	change_lr = LearningRateScheduler(scheduler)
	change_lw = LossWeightsModifier(alpha, beta, gamma)
	cbks = [change_lr, tb_cb, change_lw]

	model.summary()

	# ==================== Train CNN Model  ====================

	# Fit the model
	print("Start Training")

	model.fit(	X_train, [Y_c1_train, Y_c2_train, Y_train],
				batch_size=batch_size,
				epochs=epochs,
				verbose=1,
				callbacks=cbks,
				validation_data=(X_test, [Y_c1_test, Y_c2_test, Y_test]))

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

	jsonfile = MODEL_fILE
	hdf5file = WEIGHT_fILE

	# serialize model to JSON
	model_json = model.to_json()
	with open(jsonfile, "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights(hdf5file)
	print("Hi_fashion Model saved")
	print('===============================\n')


########## trainer trains model #########
trainer(batch_size, epochs, test_size, DATA_DIR, LOG_DIR, MODEL_FILE, WEIGHT_fILE)
