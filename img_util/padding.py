''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
padding.py
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
out_dir = '../data/leedsbutterfly/padded'
seg_dir = '../data/leedsbutterfly/segmentations'

IMG_DIR = here(img_dir)
OUT_DIR = here(out_dir)
SEG_DIR = here(seg_dir)

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Pad the image with 0 to the desired size

Input: image_path, h, w (size)
Output: padded_image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def pad(img_path, h, w):
	img = cv2.imread(img_path)
	if img is None:
		print ('Error opening image!')
		print ('Usage: copy_make_border.py [image_name -- default ../data/lena.jpg] \n')
		return -1
	borderType = cv2.BORDER_CONSTANT
	value = 0

	img_h = img.shape[0]
	img_w = img.shape[1]
	#print("image size: %d, %d" %(img_h, img_w))
	margin_h = (h - img_h)/2
	margin_w = (w - img_w)/2

	top = math.ceil(margin_h)
	bottom = math.floor(margin_h)
	left = math.ceil(margin_w)
	right = math.floor(margin_w)

	pad_out = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)

	#cv2.imshow('img',img)
	#cv2.imshow('padded',pad_out)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return pad_out

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Find the max height and width amont images in the directory

Input: img_dir
Output: height, width
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def pad_size(img_dir):
	max_h = 0
	max_w = 0
	h_name = ''
	w_name = ''
	for img in os.listdir(img_dir):
		img_name = os.fsdecode(img)
		if img_name.endswith(".png"):
			im = cv2.imread(img_dir + '/' + img_name, -1)
			#h, w = im.shape[:2]
			h = im.shape[0]
			w = im.shape[1]
			if h > max_h:
				max_h = h
				h_name = img_name
			if w > max_w:
				max_w = w
				w_name = img_name
	return max_h, max_w, h_name, w_name

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Pad each image file in img_dir 
given the desired size h, w 
output the padded image to out_dir

Input: img_dir, out_dir, h, w
Output: padded_images
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def pad_files(img_dir, out_dir, h, w):
	for img in os.listdir(img_dir):
		img_name = os.fsdecode(img)
		if img_name.endswith("_s.png"):
			img_id = img_name[:-6]
			pad_out = pad(img_dir + '/' + img_name, h, w)
			filename = img_id + '_p' + '.png'
			cv2.imwrite(pathjoin(out_dir, filename) ,pad_out)
			print(pathjoin(out_dir, filename))
	return None

#img_path = IMG_DIR + '/0010002_s.png'
h, w, h_name, w_name = pad_size(SEG_DIR)
#print("pad size is %d, %d, h_max from %s, w_max from %s" %(h,w, h_name, w_name))
#pad(img_path, h, w)
pad_files(IMG_DIR, OUT_DIR, h, w)
