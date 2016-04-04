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


#GT_PATH = "ISBI2016_ISIC_Part2_Training_GroundTruth/"
TRAIN_PATH = "ISBI2016_ISIC_Part2_Training_Data/"
TEST_PATH = "/home/ubuntu/data/SegNet2/Fextract/ISBI/TeData/"
GT_PATH = "/home/ubuntu/data/SegNet2/Fextract/predicted_superpixels/"

train_files = os.listdir(TEST_PATH)
im_files = [item[:12] for item in train_files]

for file in im_files:
	image_sup = Image.open(TEST_PATH+file+'_superpixels.png')
	image_sup = np.array(image_sup)
	image_sup = decodeSuperpixelIndex(image_sup)
	raw_gt = json.load(open(GT_PATH+file+".json"))
	globules_gt = np.array(raw_gt["globules"])
	streaks_gt = np.array(raw_gt["streaks"])
	if len(np.unique(image_sup)) != len(globules_gt):
		print file,len(np.unique(image_sup)),len(globules_gt),len(streaks_gt)

