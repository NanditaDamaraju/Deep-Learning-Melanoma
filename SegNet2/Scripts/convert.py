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


GT_PATH = "ISBI2016_ISIC_Part2_Training_GroundTruth/"
TRAIN_PATH = "ISBI2016_ISIC_Part2_Training_Data/"
NEW_GT_PATH = "new_ground_truth/"

train_files = os.listdir(TRAIN_PATH)
im_files = [item[:12] for item in train_files]

for file in im_files:
	image = Image.open(TRAIN_PATH+file+'.jpg')
	image = np.array(image)

	image_sup = Image.open(TRAIN_PATH+file+'_superpixels.png')
	image_sup = np.array(image_sup)
	image_sup = decodeSuperpixelIndex(image_sup)

	raw_gt = json.load(open(GT_PATH+file+".json"))
	globules_gt = np.array(raw_gt["globules"])
	streaks_gt = np.array(raw_gt["streaks"])

	gt_sup = []
	gt = np.zeros((image.shape[0],image.shape[1]))
	for superpixel in range(int(image_sup.min()),int(image_sup.max())+1):
		if (globules_gt[superpixel]==1 and streaks_gt[superpixel]==1):
			gt_sup.append(3)
		elif (globules_gt[superpixel]==1 and streaks_gt[superpixel]==0):
			gt_sup.append(1)
		elif (globules_gt[superpixel]==0 and streaks_gt[superpixel]==1):
			gt_sup.append(2)
		else:
			gt_sup.append(0)
		gt[image_sup==superpixel] = gt_sup[superpixel]

	result = Image.fromarray(gt.astype(np.uint8))
	result.save(NEW_GT_PATH+file+"_gt.jpg")

