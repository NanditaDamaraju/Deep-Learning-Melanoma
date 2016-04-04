import os
import random

PATH_TRAIN = "/home/ubuntu/data/SegNet/ISBI/train_all/transformed/"
PATH_TRAIN_LABEL = "/home/ubuntu/data/SegNet/ISBI/trainannot_all/transformed/"

#PATH_TEST = "/home/sahbi/Projects/SegNet/ISBI/test/"
#PATH_TEST_LABEL = "/home/sahbi/Projects/SegNet/ISBI/testannot/"


files = os.listdir(PATH_TRAIN)
random.shuffle(files)

files_train = files[0:len(files)*4//5]
files_test = files[len(files)*4//5:len(files)]


# generates train.txt
with open("train_supplemented.txt","w") as f:
	for file in files_train:
		f.write(PATH_TRAIN+file +"\t"+PATH_TRAIN_LABEL+file.split('.')[0]+"_Segmentation.png"+"\n")
		print PATH_TRAIN+file +"\t"+PATH_TRAIN_LABEL+file.split('.')[0]+"_Segmentation.png"

# generates test.txt
with open("test_supplemented.txt","w") as f:
	for file in files_test:
                f.write(PATH_TRAIN+file +"\t"+PATH_TRAIN_LABEL+file.split('.')[0]+"_Segmentation.png"+"\n")
                print PATH_TRAIN+file +"\t"+PATH_TRAIN_LABEL+file.split('.')[0]+"_Segmentation.png"
