''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
scale_pad.py
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import sys
import os
import numpy as np
import cv2
import math

''' 
Set relative path to absolute
'''
here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
pathjoin = os.path.join

img_dir = '../data/leedsbutterfly/segmented'
out_dir = '../data/leedsbutterfly/scale-leeds-butterfly-180'

IMG_DIR = here(img_dir)
OUT_DIR = here(out_dir)

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Scale and pad the image to a desired squre size

Input: image_path, size
Output: scaled_padded_image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def scale_pad(img_path, size):
	img = cv2.imread(img_path)
	if img is None:
		print ('Error opening image!')
		print ('Usage: copy_make_border.py [image_name -- default ../data/lena.jpg] \n')
		return -1
	ori_size = img.shape[:2]	# (h,w)
	ratio = float(size)/max(ori_size)
	scale_size = tuple([int(x * ratio) for x in ori_size])

	img = cv2.resize(img, (scale_size[1], scale_size[0]))

	h_diff = size - scale_size[0]
	w_diff = size - scale_size[1]

	top, bottom = h_diff//2, h_diff-(h_diff//2)
	left, right = w_diff//2, w_diff-(w_diff//2)

	scale_pad_out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)

	#cv2.imshow('img',img)
	#cv2.imshow('padded',pad_out)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return scale_pad_out

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Scale the image to a desired squre size without preserving ratio

Input: image_path, size
Output: scaled_image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def scale(img_path, size):
	img = cv2.imread(img_path)
	if img is None:
		print ('Error opening image!')
		print ('Usage: copy_make_border.py [image_name -- default ../data/lena.jpg] \n')
		return -1
	out = cv2.resize(img, (size,size))

	#cv2.imshow('img',img)
	#cv2.imshow('padded',pad_out)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return out


''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Scale and pad each image file in img_dir 
given the desired size 
output the scaled_padded image to out_dir

Input: img_dir, out_dir, size
Output: scaled_padded_images
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def scale_pad_files(img_dir, out_dir, size):
	for img in os.listdir(img_dir):
		img_name = os.fsdecode(img)
		if img_name.endswith("_s.png"):
			img_id = img_name[:-6]
			sp_out = scale_pad(img_dir + '/' + img_name, size)
			filename = img_id + '_' + str(size) + '.png'
			cv2.imwrite(pathjoin(out_dir, filename) ,sp_out)
			print(pathjoin(out_dir, filename))
	return None

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Scale each image file in img_dir 
given the desired square size 
output the scaled image to out_dir

Input: img_dir, out_dir, size
Output: scaled_images
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def scale_files(img_dir, out_dir, size):
	for img in os.listdir(img_dir):
		img_name = os.fsdecode(img)
		if img_name.endswith("_s.png"):
			img_id = img_name[:-6]
			sp_out = scale(img_dir + '/' + img_name, size)
			filename = img_id + '_' + str(size) + '.png'
			cv2.imwrite(pathjoin(out_dir, filename) ,sp_out)
			print(pathjoin(out_dir, filename))
	return None

def scale_class_files(img_dir, out_dir, size, class_size):
	class_id = 0
	count = 0
	for img in os.listdir(img_dir):
		img_name = os.fsdecode(img)
		if img_name.endswith(".jpg"):
			count += 1
			if count > class_size:
				class_id += 1
				count = 0
			sp_out = scale_pad(img_dir + '/' + img_name, size)
			filename = str(class_id) + '_' + str(count) + '.jpg'
			cv2.imwrite(pathjoin(out_dir, filename) ,sp_out)
			print(pathjoin(out_dir, filename))
	return None



size = 180
#img_path = IMG_DIR + '/0010002_s.png'
#scale_pad(img_path, size)

#scale_pad_files(IMG_DIR, OUT_DIR, size)
scale_files(IMG_DIR, OUT_DIR, size)
