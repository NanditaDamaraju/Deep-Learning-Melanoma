from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
#theano.config.device = 'gpu0'
#theano.config.floatX = 'float32'
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

######################## Defining Deep Model #################################
model = Sequential()
train_no_layer = [
	ZeroPadding2D((1,1),input_shape=(3,224,224)),
    	Convolution2D(64, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(64, 3, 3, activation='relu'),
    	MaxPooling2D((2,2), strides=(2,2)),

    	ZeroPadding2D((1,1)),
    	Convolution2D(128, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(128, 3, 3, activation='relu'),
    	MaxPooling2D((2,2), strides=(2,2)),

    	ZeroPadding2D((1,1)),
    	Convolution2D(256, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(256, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(256, 3, 3, activation='relu'),
    	MaxPooling2D((2,2), strides=(2,2)),

    	ZeroPadding2D((1,1)),
    	Convolution2D(512, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(512, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(512, 3, 3, activation='relu'),
    	MaxPooling2D((2,2), strides=(2,2)),

    	ZeroPadding2D((1,1)),
    	Convolution2D(512, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(512, 3, 3, activation='relu'),
    	ZeroPadding2D((1,1)),
    	Convolution2D(512, 3, 3, activation='relu'),
    	MaxPooling2D((2,2), strides=(2,2)),
    ]

train_yes_layer = [
    	Flatten(),
    	Dense(4096, activation='relu'),
    	Dropout(0.5),
    	Dense(4096, activation='relu'),
    	Dropout(0.5),
    	Dense(1000, activation='softmax')
	]  	


    #return model
for l in train_no_layer + train_yes_layer:
    	model.add(l)

#for l in train_no_layer:
#	l.trainable = False

model.load_weights('/home/ubuntu/data/Phase3/Weights/vgg16_weights.h5')
   # model.load_weights()
model.layers.pop()
model.layers.pop()
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=False)
model.compile(optimizer=sgd, loss='mse')

print "Model Building Finished"    

######################### Training #####################################
model.fit(im2, TrLabel, nb_epoch=20, batch_size=8, shuffle=True, validation_split=0.1, show_accuracy=True, verbose=1)
json_string = model.to_json()
f3 = open('/home/ubuntu/data/Phase3/Model/model_architecture.json', 'w')
f3.write(json_string)
model.save_weights('/home/ubuntu/data/Phase3/Weights/my_model_weights.h5')
f3.close()
