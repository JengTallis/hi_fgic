''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plot_acc.py
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint

''' 
Set relative path to absolute
'''
here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
pathjoin = os.path.join

log_dir = '../json_log/'
plot_dir = '../plot/'

LOG_DIR = here(log_dir)
PLOT_DIR = here(plot_dir)

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Find the max height and width amont images in the directory

Input: history_file, out_dir
Output: plots
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

def plot_comp_acc(m1, m2, b, out):
	h_m1 = LOG_DIR + '/' + m1 + '.json'
	h_m2 = LOG_DIR + '/' + m2 + '.json'
	h_b = LOG_DIR + '/' + b + '.json'
	out_file = PLOT_DIR + '/' + out + '.png'

	data1 = json.load(open(h_m1))
	data2 = json.load(open(h_m2))
	data3 = json.load(open(h_b))

	acc1 = data1["val_fg_pred_acc"]
	acc2 = data2["val_fg_pred_acc"]
	acc3 = data3["val_acc"]

	n_epochs = len(acc1)
	epochs = list(range(1,n_epochs+1))
	plt.plot(epochs, acc1, 'b--', label='hi_m1 test_acc')
	plt.plot(epochs, acc2, 'g--', label='hi_m2 test_acc') 
	plt.plot(epochs, acc3, 'r--', label='base test_acc')
	plt.title('Test accuracy of different models')
	plt.legend()
	plt.savefig(out_file)	# does not support jpg
	plt.clf()
	return None



def plot_hi_model_acc(history_file, out_file):
	
	data = json.load(open(history_file))
	#pprint(data)

	# train_loss
	loss = data["loss"]
	c1_loss = data["c1_pred_loss"]
	c2_loss = data["c2_pred_loss"]
	fg_loss = data["fg_pred_loss"]

	# val_loss
	val_loss = data["val_loss"]
	c1_val_loss = data["val_c1_pred_loss"]
	c2_val_loss = data["val_c2_pred_loss"]
	fg_val_loss = data["val_fg_pred_loss"]

	# train_acc
	c1_acc = data["c1_pred_acc"]
	c2_acc = data["c2_pred_acc"]
	fg_acc = data["fg_pred_acc"]

	# test_acc
	c1_test_acc = data["val_c1_pred_acc"]
	c2_test_acc = data["val_c2_pred_acc"]
	fg_test_acc = data["val_fg_pred_acc"]

	#print(c1_test_acc)
	#print(c2_test_acc)
	#print(fg_test_acc)

	n_epochs = len(c1_acc)
	epochs = list(range(1,n_epochs+1))

	plt.plot(epochs, c1_test_acc, 'b--', label='c1 test_acc')
	plt.plot(epochs, c2_test_acc, 'g--', label='c2 test_acc') 
	plt.plot(epochs, fg_test_acc, 'r--', label='fg test_acc')

	plt.title('Test accuracy of each branch')
	plt.legend()
	#plt.figure()
	#plt.plot(epochs, loss, 'bo', label='Training loss')
	#plt.plot(epochs, val_loss, 'b', label='Validation loss')
	#plt.legend()
	#plt.title('Training and validation loss')
	#plt.show()
	plt.savefig(out_file)	# does not support jpg
	plt.clf()
	return None
name = 'hi_butterfly_m1_r128_180'



hi_files = ['20','22','23','29','30','36','37']
b_files  = ['21','24','31','38']

def plot_hi_models_acc(files):
	for name in files:
		history_file = LOG_DIR + '/' + name + '.json'
		out_file = PLOT_DIR + '/' + name + '.png'
		plot_hi_model_acc(history_file, out_file)


#plot_hi_models_acc(hi_files)

m1 = '29'
m2 = '30'
b = '31'
out = 'r128_180'
plot_comp_acc(m1, m2, b, out)


