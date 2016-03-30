from keras.models import Sequential, model_from_json
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import glob, csv
import cPickle as pickle

############################# Loading Training Data #########################

f1 = open("/home/ubuntu/data/Phase3/Parsed/TrData.txt", "rb")
im2 = pickle.load(f1)
f2 = open("/home/ubuntu/data/Phase3/Parsed/TrLabel.txt", "rb")
TrLabel = pickle.load(f2)

f1.close()
f2.close()

print "Loading Training Data Finished"

######################## Loading Deep Model #################################
f3 = open('/home/ubuntu/data/Phase3/Model/model_architecture.json')
model = model_from_json(f3.read())
model.load_weights('/home/ubuntu/data/Phase3/Weights/my_model_weights.h5')
f3.close()
score = model.evaluate(im2, TrLabel, batch_size=16)
print score
