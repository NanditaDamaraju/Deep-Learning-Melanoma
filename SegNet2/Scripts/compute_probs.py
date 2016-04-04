import os
import numpy as np
import json
from PIL import Image
from collections import Counter


NEW_GT_PATH = "/home/ubuntu/data/SegNet2/Fextract/trainannot_streaks_resized/"
train_files = os.listdir(NEW_GT_PATH)

prev = np.zeros(2)
for file in train_files:
	image = Image.open(NEW_GT_PATH+file)
	image = np.array(image)
	counts = np.bincount(image.flatten())
	if len(counts) == 1:
		counts=np.append(counts,[0.])
	#print file,counts,np.unique(image.flatten())
	prev += counts
	#print "sum",prev

print prev
print [np.median(prev)/i for i in prev]

# [  2.82841768e+01,   2.04493730e-01,   2.42114903e-02,1.00000000e-03]
