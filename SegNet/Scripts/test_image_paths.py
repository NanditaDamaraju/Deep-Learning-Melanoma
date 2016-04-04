import os

dir_name = "/home/ubuntu/data/SegNet/ISBI/test_ISBI/transformed/"

for file in os.listdir(dir_name):
	print dir_name + file + '\t' + "/home/ubuntu/data/SegNet/ISBI/trainannot_all/transformed/ISIC_0000000_Segmentation.png"
