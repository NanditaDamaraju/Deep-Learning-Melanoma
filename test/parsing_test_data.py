import cv2, numpy as np
import glob, csv
import cPickle as pickle

############################# Preparing Training Data #########################
im2 = cv2.resize(cv2.imread('/home/ubuntu/data/Phase3/Test/Data/ISIC_0000003.jpg'), (224, 224)).astype(np.float32)
im2[:,:,0] -= 103.939
im2[:,:,1] -= 116.779
im2[:,:,2] -= 123.68
im2 = im2.transpose((2,0,1))
im2 = np.expand_dims(im2, axis=0)

for name in sorted(glob.glob('/home/ubuntu/data/Phase3/Test/Data/ISIC_00?????.jpg')):
	if name=='/home/ubuntu/data/Phase3/Test/Data/ISIC_0000003.jpg':
		continue
	temp = cv2.resize(cv2.imread(name), (224, 224)).astype(np.float32)
    	temp[:,:,0] -= 103.939
    	temp[:,:,1] -= 116.779
    	temp[:,:,2] -= 123.68
    	temp = temp.transpose((2,0,1))
    	temp = np.expand_dims(temp, axis=0)
	im2 = np.concatenate((im2,temp))
#im2 stores training data, format: im2[i,:,:,:] is ith image data
#im2 size: 900x3x224x224

f1 = open("/home/ubuntu/data/Phase3/Parsed/TestData.txt", "wb")
pickle.dump(im2,f1)
f1.close()

