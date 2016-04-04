import cv2
import numpy as np
import os

data_dir = "/home/ubuntu/data/SegNet/ISBI/trainannot_all/"

for file in os.listdir(data_dir):
	filename = file.split('_S')[0]
	print data_dir + filename + "_Segmentation.png"
	image = cv2.imread(data_dir + filename + "_Segmentation.png")
	flip_v = cv2.flip(image, 0)
	flip_h = cv2.flip(image, 1)

	rotate_180 = cv2.flip(flip_h, 0)

	cv2.imwrite(data_dir + filename + '_flipvert_Segmentation.png', flip_v)
	cv2.imwrite(data_dir + filename + '_fliphorz_Segmentation.png', flip_h)
	cv2.imwrite(data_dir + filename + '_rotate180_Segmentation.png', rotate_180) 	
