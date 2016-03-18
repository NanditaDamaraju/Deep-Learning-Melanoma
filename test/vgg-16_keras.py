from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import glob, csv

############################# Preparing Training Data #########################
if __name__ == "__main__":
    im = cv2.resize(cv2.imread('Greece-Cat.jpg'), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    im2 = cv2.resize(cv2.imread('Data/ISIC_0000000.jpg'), (224, 224)).astype(np.float32)
    im2[:,:,0] -= 103.939
    im2[:,:,1] -= 116.779
    im2[:,:,2] -= 123.68
    im2 = im2.transpose((2,0,1))
    im2 = np.expand_dims(im2, axis=0)

for name in sorted(glob.glob('Data/ISIC_00?????.jpg')):
	if name=='Data/ISIC_0000000.jpg':
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
TrLabel=[]
with open('GTruth.csv') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
                if row[1]=='benign':
                        TrLabel.append([1,0])
                else:
                        TrLabel.append([0,1])

print "Preparing Training Data Finished"

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

   # model.load_weights('vgg16_weights.h5')
   # model.load_weights()

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

print "Model Building Finished"    

######################### Training #####################################
#test_set = [[0,1],[0,1]]
model.fit(im2, TrLabel, nb_epoch=10, batch_size=32)
out = model.predict(im)
out2 = model.predict(np.expand_dims(im2[0,:,:,:], axis=0))
print "catty"
print (out)
print "Cancer"
print (out2)

