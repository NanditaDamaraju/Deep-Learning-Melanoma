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
    	Dense(1000, activation='softmax'),
    	Dense(2, activation='softmax',init='uniform'),
    ]

    #return model
for l in train_no_layer + train_yes_layer:
    	model.add(l)

for l in train_no_layer:
	l.trainable = False

model.load_weights('/home/ubuntu/data/Phase3/Weights/vgg16_weights.h5')
   # model.load_weights()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

print "Model Building Finished"    

######################### Training #####################################
#test_set = [[0,1],[0,1]]
model.fit(im2, TrLabel, nb_epoch=1, batch_size=32)
#out = model.predict(im)
out2 = model.predict(np.expand_dims(im2[0,:,:,:], axis=0))
print "Cancer"
print (out2)

