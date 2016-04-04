import os
import numpy as np
import json
from PIL import Image

def decodeSuperpixelIndex(rgbValue):
    """
    Decode an RGB representation of a superpixel label into its native scalar value.
    :param pixelValue: A single pixel, or a 3-channel image.
    :type pixelValue: numpy.ndarray of uint8, with a shape [3] or [n, m, 3]
    """
    return \
        (rgbValue[..., 0].astype(np.uint64)) + \
        (rgbValue[..., 1].astype(np.uint64) << np.uint64(8)) + \
        (rgbValue[..., 2].astype(np.uint64) << np.uint64(16))


TRAIN_PATH = "/home/ubuntu/data/SegNet2/Fextract/ISBI/TeData/"
PREDICTED_GLOBULES = "/home/ubuntu/data/SegNet2/Fextract/test_globules_final/"
PREDICTED_STREAKS = "/home/ubuntu/data/SegNet2/Fextract/test_streaks_final/"
NEW_PREDICTED = "/home/ubuntu/data/SegNet2/Fextract/predicted_superpixels/"

pred_files = os.listdir(PREDICTED_GLOBULES)
existing = os.listdir(NEW_PREDICTED)
im_files = [item[:12] for item in pred_files if (item[:12]+'.json') not in existing]
for file in im_files:
	print file
	image_globules = Image.open(PREDICTED_GLOBULES+file+'_Segmentation.png')
	image_streaks = Image.open(PREDICTED_STREAKS+file+'_Segmentation.png')
	image_globules = np.array(image_globules)
	image_streaks = np.array(image_streaks)
	image_sup = Image.open(TRAIN_PATH+file+'_superpixels.png')
	image_sup = np.array(image_sup)
	image_sup = decodeSuperpixelIndex(image_sup)
	result = {"globules":[],"streaks":[]}
	for superpixel in range(int(image_sup.min()),int(image_sup.max())+1):
		a = image_globules[image_sup==superpixel]
		b = image_streaks[image_sup==superpixel]
		#print np.unique(a), np.unique(b)
		classeA = float(sum(map(sum, a)))/(a.size*255)
		classeB = float(sum(map(sum, b)))/(b.size*255)
		result["globules"].append(classeA)		
		result["streaks"].append(classeB)
	with open(NEW_PREDICTED+file+".json", 'w') as outfile:
		json.dump(result, outfile)
	print "end"
