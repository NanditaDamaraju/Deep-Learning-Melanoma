import os

PATH_TRAIN = "/home/sahbi/Projects/SegNet/ISBI/train/"
PATH_TRAIN_LABEL = "/home/sahbi/Projects/SegNet/ISBI/trainannot/"

PATH_TEST = "/home/sahbi/Projects/SegNet/ISBI/test/"
PATH_TEST_LABEL = "/home/sahbi/Projects/SegNet/ISBI/testannot/"

# generates train.txt
with open("train.txt","w") as file:
	for files in os.listdir("train"):
		file.write(PATH_TRAIN+files+" "+PATH_TRAIN_LABEL+files[0:12]+"_Segmentation.png"+"\n")
		print PATH_TRAIN+files+" "+PATH_TRAIN_LABEL+files[0:12]+"_Segmentation.png"

# generates test.txt
with open("test.txt","w") as file:
	for files in os.listdir("test"):
		file.write(PATH_TEST+files+" "+PATH_TEST_LABEL+files[0:12]+"_Segmentation.png"+"\n")
		print PATH_TEST+files+" "+PATH_TEST_LABEL+files[0:12]+"_Segmentation.png"

