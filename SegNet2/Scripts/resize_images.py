import os
import cv2

path = '/home/ubuntu/data/SegNet2/Fextract/test_resized'
for file in os.listdir(path):
	image = cv2.imread(path+"/"+file)
	if image is not None: 
		print file
		#print image[image>0]	
		resized_image = cv2.resize(image, (256, 192))
		cv2.imwrite(path + '_resized/' + file, resized_image)
