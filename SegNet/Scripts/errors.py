import os

PATH2SEGNET = "/home/ubuntu/DL8803"
n_images = 180


# os.system("./caffe train -gpu 0 -solver "+PATH2SEGNET+"/SegNet/Models/segnet_solver.prototxt -weights "+ PATH2SEGNET +"/SegNet/Models/VGG_ILSVRC_16_layers.caffemodel")

for this_iter in range(10,410,10):
	print this_iter
	this_model = "segnet_iter_"+str(this_iter)+".caffemodel"
	command = "python "+PATH2SEGNET+"/SegNet/Scripts/compute_bn_statistics.py "+PATH2SEGNET+"/SegNet/Models/segnet_train.prototxt /home/ubuntu/data/SegNet/Models/Training/"+this_model+" /home/ubuntu/data/SegNet/Models/Inference/"
	# print command
	os.system(command)
	command2 = "python "+PATH2SEGNET+"/SegNet/Scripts/compute_test_error.py --model "+PATH2SEGNET+"/SegNet/Models/segnet_inference.prototxt --weights /home/ubuntu/data/SegNet/Models/Inference/test_weights.caffemodel --iter "+str(n_images)
	# print command2
	os.system(command2)
