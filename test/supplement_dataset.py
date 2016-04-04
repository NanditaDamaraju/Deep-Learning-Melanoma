import cv2
import numpy as np
import os

data_dir = "/home/ubuntu/data/Phase3/Data/"
gt_path = "/home/ubuntu/data/Phase3/GT/GTruth.csv"

open("/home/ubuntu/data/Phase3/GT/GT_supplement.csv", 'w').close()

dict = {}
for line in open(gt_path):	
	filename, val = line.split(',')
	dict[filename] = val.strip() 

file_list = os.listdir(data_dir)
for file in file_list:	
	id = file.split('.')[0]
	with open("/home/ubuntu/data/Phase3/GT/GT_supplement.csv", 'a') as f:
        	f.write(id + ',' + dict[id])

	if(dict[id] == 'malignant'):
		print data_dir + file
		image = cv2.imread(data_dir + file)

		flip_v = cv2.flip(image, 0)
		flip_h = cv2.flip(image, 1)

		rotate_180 = cv2.flip(flip_h, 0)

		cv2.imwrite(data_dir + id + '_flipvert.jpg', flip_v)
		cv2.imwrite(data_dir + id + '_fliphorz.jpg', flip_h)
		cv2.imwrite(data_dir + id + '_rotate180.jpg', rotate_180) 	

		with open("/home/ubuntu/data/Phase3/GT/GT_supplement.csv", 'a') as f:
			f.write(id + '_flipvert,malignant\n')
			f.write(id + '_fliphorz,malignant\n')
			f.write(id + '_rotate180,malignant\n')
