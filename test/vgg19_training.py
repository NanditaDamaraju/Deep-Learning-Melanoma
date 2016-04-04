from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
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

#print len(im2), len(im2v), len(TrLabel), len(VLabel)
print "Loading Training and Validation Data Finished"

######################## Defining Deep Model #################################
model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))


#model.load_weights('/home/ubuntu/data/Phase3/Weights/vgg19_weights.h5')
   # model.load_weights()
model.layers.pop()
model.layers.pop()
model.add(Dense(2, activation='softmax'))
model.load_weights('/home/ubuntu/data/Phase3/Weights/weights_best_2.hdf5')
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.1, nesterov=False)
model.compile(optimizer=sgd, loss='mse')
#model.load_weights('/home/ubuntu/data/Phase3/Weights/weights_best.h5')
print "Model Building Finished"    

######################### Training #####################################

checkpointer = ModelCheckpoint(filepath="/home/ubuntu/data/Phase3/Weights/weights_best_2.hdf5", verbose=1, save_best_only=True)
model.fit(im2, TrLabel, nb_epoch=20, batch_size=8, shuffle=True, validation_split=0.2,	 show_accuracy=True, verbose=1, callbacks=[checkpointer])
json_string = model.to_json()

f3 = open('/home/ubuntu/data/Phase3/Model/model_architecture_vgg19.json', 'w')
f3.write(json_string)
model.save_weights('/home/ubuntu/data/Phase3/Weights/my_model_weights_vgg19.h5')
f3.close()

