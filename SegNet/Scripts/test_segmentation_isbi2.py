import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import operator
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
from sklearn.metrics import jaccard_similarity_score
caffe_root = '/home/ubuntu/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

scores=[]

for i in range(0, args.iter):
#for i in range(0,1):
	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	predicted2 = net.blobs['conv1_1_D'].data
	print predicted.shape
	print predicted2.shape	
	xshape = 192
	yshape = 256
	kernal = 2
	image = np.squeeze(image[0,:,:,:])
	label = np.squeeze(label[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)
	y  = [[0] * xshape for i in range(yshape)]
	output3 = np.squeeze(predicted2[0,1,:,:])
	output3 = [[(x - x) for x in row] for row in output3]
	for mindex in range(1,kernal):
		output2 = np.squeeze(predicted2[0,mindex,:,:])	
		xyz = zip(*output2)
		mymax = max(map(max,xyz))
		mymin = min(map(min,xyz))
		xyz = zip(*output2)
		mymax = max(map(max,xyz))
		mymin = min(map(min,xyz))
		output2 = [[(x - mymin) for x in row] for row in output2]
		if mymax != 0:
	        	output2 = [[(x/mymax) for x in row] for row in output2]
        	output2 = [[int(math.floor(x*255)) for x in row] for row in output2]
		for i1 in range(1,xshape):
			for j1 in range(1,yshape):
				output3[i1][j1]+=output2[i1][j1]



#		output3 = [x + y for x, y in zip(output3, output2)]
#		output3 = map(operator.add, output3, output2)
	#print max(output2.any())
#	for i in range(1,192):
#		for j in range(1,256):
#			print output2[i,j]
#
#

#	xyz = zip(*output2)
#	mymax = max(map(max,xyz))
#	mymin = min(map(min,xyz))
#	#output2[:,:] = [x - mymin for x in output2,x - mymin for x in output2] 
#	output2 = [[(x - mymin) for x in row] for row in output2]
#	output2 = [[(x/mymax) for x in row] for row in output2]
	output3 = [[int(math.floor(x/kernal)) for x in row] for row in output3]
	print output2

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()
	r2 = output3
	g2 = output3
	b2 = output3

	Skin = [0,0,0]
	Lesion = [255,255,255]

	label_colours = np.array([Skin,Lesion])
	for l in range(0,2):
		r[ind==l] = label_colours[l,0]
		g[ind==l] = label_colours[l,1]
		b[ind==l] = label_colours[l,2]
		r_gt[label==l] = label_colours[l,0]
		g_gt[label==l] = label_colours[l,1]
		b_gt[label==l] = label_colours[l,2]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = r/255.0
	rgb[:,:,1] = g/255.0
	rgb[:,:,2] = b/255.0
	rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb_gt[:,:,0] = r_gt/255.0
	rgb_gt[:,:,1] = g_gt/255.0
	rgb_gt[:,:,2] = b_gt/255.0
	rgb2 = np.zeros((xshape, yshape, 3))
	rgb2[:,:,0] = r2
	rgb2[:,:,1] = g2
	rgb2[:,:,2] = b2
	image = image/255.0

	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]

	# print image.shape,rgb_gt.shape,rgb.shape
	#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save(IMAGE_FILE+'_segnet.png')

	plt.figure()
	plt.imsave("out1.png", image, vmin=0, vmax=1)
	plt.figure()
	plt.imsave("out2.png", rgb_gt,vmin=0, vmax=1)
	plt.figure()
	plt.imsave("out3.png", rgb, vmin=0, vmax=1)
	plt.figure()
	plt.imsave("out4.png", rgb2)
	#plt.show()

	y_true = label.flatten()
	y_pred = ind.flatten()

	score = jaccard_similarity_score(y_true, y_pred)
	scores.append(score)
	print "image ",i," score: ",score


print 'Success!'
print "mean accuracy: ",np.array(scores).mean()
