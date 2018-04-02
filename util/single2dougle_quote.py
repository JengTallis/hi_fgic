''' 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Transform single quote dictionary into double quote json format

Input: dir
Output: files in json format
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import os
import json

here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
pathjoin = os.path.join

log_dir = '../log/'

json_log_dir = '../json_log/'
LOG_DIR = here(log_dir)
JSON_LOG_DIR = here(json_log_dir)

def data_as_json(file):
	data = None
	with open(file,'r') as f:
		file = f.readlines()
		temp = str(file).replace("'", '"')
		data = temp[2:-2]
	return data

def file2json(file, outfile):
	data = data_as_json(file)
	with open(outfile, 'w') as f:
		f.write(data)


def json_files(file_names):
	for file_name in file_names:
		history_file = LOG_DIR + '/' + file_name + '.txt'
		json_file = JSON_LOG_DIR + '/' + file_name + '.json'
		file2json(history_file, json_file)


files = ['20','21','22','23','24','29','30','31','36','37','38']
json_files(files)

