''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
hi_leeds_butterfly.py

Run:
floyd run \
--cpu2 \
--data jengtallis/datasets/scale-leeds-butterfly-128/1:/data \
"python hi_leeds_butterfly.py"


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import keras
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

class LossWeightsModifier2(keras.callbacks.Callback):
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
    if epoch == 58:
      be.set_value(self.alpha, 0.98)
      be.set_value(self.beta, 0.01)
      be.set_value(self.gamma, 0.01)
    if epoch == 68:
      be.set_value(self.alpha, 0.1)
      be.set_value(self.beta, 0.8)
      be.set_value(self.gamma, 0.1)
    if epoch == 78:
      be.set_value(self.alpha, 0)
      be.set_value(self.beta, 0)
      be.set_value(self.gamma, 1)

class LossWeightsModifier3(keras.callbacks.Callback):
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
    if epoch == 68:
      be.set_value(self.alpha, 0.98)
      be.set_value(self.beta, 0.01)
      be.set_value(self.gamma, 0.01)
    if epoch == 78:
      be.set_value(self.alpha, 0.1)
      be.set_value(self.beta, 0.8)
      be.set_value(self.gamma, 0.1)
    if epoch == 88:
      be.set_value(self.alpha, 0)
      be.set_value(self.beta, 0)
      be.set_value(self.gamma, 1)   

'''
########################################################
# ================== Data Directory  ===================
''' 

# Image Directory
IMG_DIR = '/data'

# Output File Path
LOG_DIR = '/output/tb_log/'
output_dir = '/output/'
train_id = '1'
model_name = 'model_hi_leedsbutterfly_' + train_id + '.json'
weight_name = 'weights_hi_leedsbutterfly_' + train_id + '.h5'
MODEL_FILE = os.path.join(output_dir, model_name)
WEIGHT_FILE = os.path.join(output_dir, weight_name)
historyfile = '/output/hi_butterfly_his_'+ train_id + '.txt'

# === Input/Output data ===
def data(img_dir, test_size):
	a = []
	b = []
	for img in os.listdir(img_dir):
		img_name = os.fsdecode(img)
		str_class = img_name[:-12]
		int_class = int(str_class) - 1
		if img_name.endswith(".png"):
			im = cv2.imread(img_dir + '/' + img_name, -1)
			if im is None:
				print ('Error opening image!')
				return -1
			a.append(im)
			b.append(int_class)

	X = np.array(a)
	Y = np.array(b)

	Y = keras.utils.to_categorical(Y)
	X = X.astype('float32')
	X = (X - np.mean(X)) / np.std(X)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

	return X_train, X_test, Y_train, Y_test



'''
#########################################################
# ================= Model Parameters  ===================
'''
batch_size	= 128
epochs		= 120

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
	scores = model.evaluate(X, [Y_c1_test, Y_c2_test, Y_test], verbose=0)
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))
	p = model.predict(X[0])
	print(p)


'''
########################################################
# ==================== CNN Trainer  ====================
'''
def trainer(batch_size, epochs, test_size, IMG_DIR, LOG_DIR, MODEL_FILE, WEIGHT_FILE):

	# ==================== Data  ====================

	# ==================== data definition =====================
	size = 128
	input_shape = (size, size, 3)

	# === coarse 1 classes ===
	n_c1 = 2
	# === coarse 2 classes ===
	n_c2 = 8
	# === fine-grained classes ===
	n_fg = 10

	# Training data: Input (X) and Expected Output (Y)
	X_train, X_test, Y_train, Y_test =  data(IMG_DIR, test_size)
	print("Shape of X_train")
	print(X_train.shape)

	# ============== coarse 2 labels ===============
	fg_parent = {
		0:0,
		1:1, 2:1,
		3:2,
		4:3,
		5:4,
		6:5,
		7:6,
		8:7, 9:7
	}

	Y_c2_train = np.zeros((Y_train.shape[0], n_c2)).astype("float32")
	for i in range(Y_c2_train.shape[0]):
		Y_c2_train[i][fg_parent[np.argmax(Y_train[i])]] = 1.0

	Y_c2_test = np.zeros((Y_test.shape[0], n_c2)).astype("float32")
	for i in range(Y_c2_test.shape[0]):
		Y_c2_test[i][fg_parent[np.argmax(Y_test[i])]] = 1.0

	# ============== coarse 1 labels ===============
	c2_parent = {
		0:0, 1:0, 2:0,
		3:1, 4:1, 5:1, 6:1,
		7:0
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
	c1_bch = Dense(256, activation='relu', name='c1_fc1')(c1_bch)
	c1_bch = BatchNormalization()(c1_bch)
	c1_bch = Dropout(0.5)(c1_bch)
	c1_bch = Dense(256, activation='relu', name='c1_fc2')(c1_bch)
	c1_bch = BatchNormalization()(c1_bch)
	c1_bch = Dropout(0.5)(c1_bch)
	c1_pred = Dense(n_c1, activation='softmax', name='c1_pred')(c1_bch)

	# ==================== Block 3  ====================
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = BatchNormalization()(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# ==================== Coarse 2 Branch  ====================
	c2_bch = Flatten(name='c2_flatten')(x)
	c2_bch = Dense(512, activation='relu', name='c2_fc1')(c2_bch)
	c2_bch = BatchNormalization()(c2_bch)
	c2_bch = Dropout(0.5)(c2_bch)
	c2_bch = Dense(512, activation='relu', name='c2_fc2')(c2_bch)
	c2_bch = BatchNormalization()(c2_bch)
	c2_bch = Dropout(0.5)(c2_bch)
	c2_pred = Dense(n_c2, activation='softmax', name='c2_pred')(c2_bch)

	# ==================== Block 4  ====================
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = BatchNormalization()(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = BatchNormalization()(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# ==================== Fine-Grained Block  ====================
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	fg_pred = Dense(n_fg, activation='softmax', name='fg_pred')(x)

	model = Model(input=input_imgs, output=[c1_pred, c2_pred, fg_pred], name='hi_leeds_butterfly')

	# ==================== Compile Model  ====================
	sgd = optimizers.SGD(lr=0.003, momentum=0.9, nesterov=True)
	model.compile(	loss='categorical_crossentropy', 
					optimizer=sgd, 
					loss_weights=[alpha, beta, gamma],
					# optimizer=keras.optimizers.Adadelta(),
					metrics=['accuracy'])

	tb_cb = TensorBoard(log_dir=LOG_DIR, histogram_freq=0)
	change_lr = LearningRateScheduler(scheduler)
	change_lw = LossWeightsModifier3(alpha, beta, gamma)
	cbks = [change_lr, tb_cb, change_lw]

	model.summary()

	# ==================== Train CNN Model  ====================

	# Fit the model
	print("Start Training")

	train = model.fit(	X_train, [Y_c1_train, Y_c2_train, Y_train],
				batch_size=batch_size,
				epochs=epochs,
				verbose=1,
				callbacks=cbks,
				validation_data=(X_test, [Y_c1_test, Y_c2_test, Y_test]))

	print("Finish Training")

	# evaluate the model
	print("Evaluation Result:")
	scores = model.evaluate(X_test, [Y_c1_test, Y_c2_test, Y_test], verbose=1)
	print(scores)
	#print('Test loss:', scores[0])
	#print('Test accuracy:', scores[1])
	#print("\n%s: %.5f%%" % (model.metrics_names[1], scores[1]*100))

	# ==================== Plot result  ====================
	print('History:')
	print((str(train.history)))
	#accuracy = train.history['acc']
	#val_accuracy = train.history['val_acc']
	#loss = train.history['loss']
	#val_loss = train.history['val_loss']
	#epochs = range(len(accuracy))
	

	#plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
	#plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
	#plt.title('Training and validation accuracy')
	#plt.legend()
	#plt.figure()
	#plt.plot(epochs, loss, 'bo', label='Training loss')
	#plt.plot(epochs, val_loss, 'b', label='Validation loss')
	#plt.title('Training and validation loss')
	#plt.legend()
	#plt.show()

	
	with open(historyfile, "w") as his_file:
		his_file.write(str(train.history))
	


	# ==================== Save CNN Model  ====================

	jsonfile = MODEL_FILE
	hdf5file = WEIGHT_FILE

	# serialize model to JSON
	model_json = model.to_json()
	with open(jsonfile, "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights(hdf5file)
	print("Hi_leeds_butterfly Model saved")
	print('===============================\n')


########## trainer trains model #########
trainer(batch_size, epochs, test_size, IMG_DIR, LOG_DIR, MODEL_FILE, WEIGHT_FILE)
