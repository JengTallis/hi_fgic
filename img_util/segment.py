''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
segment.py
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import sys
import os
import numpy as np
import cv2


''' 
Set relative path to absolute
'''
here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
pathjoin = os.path.join


img_dir = '../data/leedsbutterfly/images'
seg_dir = '../data/leedsbutterfly/segmentations'
out_dir = '../data/leedsbutterfly/segmented'
IMG_DIR = here(img_dir)
SEG_DIR = here(seg_dir)
OUT_DIR = here(out_dir)


''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Obtain the segmented image given the image and the segmentation mask

Input: image (RGB), segmentation mask (Binary)
Output: segmented_image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def segment(img_path, seg_path):

	img = cv2.imread(img_path, 1)	# 1: cv2.IMREAD_COLOR
	seg = cv2.imread(seg_path, 0)	# 0: cv2.IMREAD_GRAYSCALE

	mask = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)	#change mask to a 3 channel image 
	mask_out = cv2.subtract(mask, img)
	mask_out = cv2.subtract(mask, mask_out)

	#cv2.imshow('image',img)
	#cv2.imshow('segmented', mask_out)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return mask_out

''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Segment each 
image file in img_dir with the corresponding 
segmentation mask in seg_dir and 
output the segmented image to out_dir

Input: img_dir, seg_dir, out_dir
Output: segmented_images
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
def segment_files(img_dir, seg_dir, out_dir):
	img_d = os.fsencode(img_dir)
	print(img_d)
	for img in os.listdir(img_dir):
		img_name = os.fsdecode(img)
		if img_name.endswith(".png"):
			img_id = img_name[:-4]
			seg_out = segment(img_dir + '/' + img_name, seg_dir + '/' + img_id + '_seg0.png')
			filename = img_id + '_s' + '.png'
			cv2.imwrite(pathjoin(out_dir, filename) ,seg_out)
			print(pathjoin(out_dir, filename))
	return None


#img_path = './images/0010001.png'
#seg_path = './segmentations/0010001_seg0.png'

segment_files(IMG_DIR, SEG_DIR, OUT_DIR)

