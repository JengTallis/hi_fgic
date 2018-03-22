# box_crop_scale.py

import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import ImageOps
import numpy as np
from sklearn.model_selection import train_test_split


def box_crop(img_path, annot_path): # annotation in XML file
	tree = ET.parse(annot_path)
	root = tree.getroot()
	xmin = int(root[5][4].find('xmin').text)
	ymin = int(root[5][4].find('ymin').text)
	xmax = int(root[5][4].find('xmax').text)
	ymax = int(root[5][4].find('ymax').text)
	box = (xmin, ymin, xmax, ymax) # (upper, left, lower, right)
	img = Image.open(img_path)
	box_crop_img = img.crop(box)
	#box_crop_img.show()
	return box_crop_img

img_path = 'n02085620_7.jpg'
annot_path = 'n02085620_7'
box_crop(img_path, annot_path)