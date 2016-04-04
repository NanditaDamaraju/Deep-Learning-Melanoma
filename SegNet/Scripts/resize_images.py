import os
import cv2

path = '/home/ubuntu/data/SegNet/ISBI/trainannot_all/'
for file in os.listdir(path):
	image = cv2.imread(path+file)
	if image is not None: 
		print file
		#print image[image>0]	
		resized_image = cv2.resize(image, (256, 192))
		ret, bW = cv2.threshold(resized_image,0,1,cv2.THRESH_BINARY)
		cv2.imwrite(path + 'transformed/' + file, bW)
