''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
segment.py

Obtain the segmented image given the image and the segmentation mask

Input: image (RGB), segmentation mask (Binary)
Output: segmented_image

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import sys
import numpy as np
import cv2

#img_path = './images/0010001.png'
#seg_path = './segmentations/0010001_seg0.png'

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

