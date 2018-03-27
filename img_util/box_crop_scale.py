# box_crop_scale.py

import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import ImageOps
import numpy as np
from sklearn.model_selection import train_test_split

''' 
Set relative path to absolute
'''
here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
pathjoin = os.path.join

img_dir = '../data/stanford_dogs/dogs-12'
annot_dir = '../data/stanford_dogs/annotation-12'
out_dir = '../data/stanford_dogs/box-dogs-12'

IMG_DIR = here(img_dir)
ANNOT_DIR = here(annot_dir)
OUT_DIR = here(out_dir)

def box_crop(img_path, annot_path): # annotation in XML file
	#print(img_path)
	#print(annot_path)
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


def box_crop_files(img_dir, annot_dir, out_dir):
	class_id = -2
	for d in os.listdir(img_dir):
		class_id += 1
		class_name = os.fsdecode(d)
		#print(class_id)
		#print(class_name)
		class_dir = img_dir + '/' + str(class_name)
		if class_id > -1:
			for img in os.listdir(class_dir): # d: #os.listdir(class_dir):
				img_name = os.fsdecode(img)
				if img_name.endswith(".jpg"):
					img_id = img_name[:-4]
					img_path = img_dir + '/' + class_name + '/' + img_name
					annot_path = annot_dir + '/' + class_name + '/' + img_id
					out = box_crop(img_path, annot_path)
					filename = str(class_id) + '_' + img_id + '_b' + '.jpg'
					out.save(pathjoin(out_dir, filename))
					print(pathjoin(out_dir, filename))
	return None

box_crop_files(IMG_DIR, ANNOT_DIR, OUT_DIR)


#img_path = 'n02085620_7.jpg'
#annot_path = 'n02085620_7'
#box_crop(img_path, annot_path)