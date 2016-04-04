import numpy as np
import cv2
import os
from sklearn.metrics import jaccard_similarity_score

#gt_dir = "/home/ubuntu/data/SegNet/ISBI/trainannot_all/ISIC_0000009_Segmentation.png"
test_dir = "/home/ubuntu/data/SegNet2/Fextract/test_globules_predicted/"
size_dir = "/home/ubuntu/data/SegNet2/Fextract/testannot_globules_resized/"
paths = os.listdir(test_dir)
#print paths[1]
#get labels for ground truth files
scores =[]
scores_original =[]
for path in paths:

	#image_gt = cv2.imread(gt_dir)
	#image_gt_resized = cv2.resize(image_gt, (256,192))

	#label_gt = np.sum(image_gt, axis = 2)/765
	#label_gt_resized = np.sum(image_gt_resized, axis =2)/765
	image_size = cv2.imread(size_dir + path)
	image_test = cv2.imread(test_dir + path)

	#filtered_image = cv2.GaussianBlur(image_test, (5,5),0)
	#ret, filtered_image = cv2.threshold(filtered_image, 0,255,cv2.THRESH_BINARY) 

	image_test_resized = cv2.resize(image_test, (image_size.shape[1], image_size.shape[0]))
	
	#cv2.imwrite("/home/ubuntu/" + path + "g.png",  image_gt)
	#cv2.imwrite("/home/ubuntu/" + path + "gr.png", image_gt_resized)
	#cv2.imwrite("/home/ubuntu/" + path + "bound.png", filtered_image)
	#cv2.imwrite("/home/ubuntu/" + path + "t.png", image_test)
	cv2.imwrite("/home/ubuntu/data/SegNet2/Fextract/test_globules_final/" + path.split('.')[0] + '_Segmentation.png' , image_test_resized)
	
	label_test = np.sum(image_test, axis = 2)/765
	label_test_resized = np.sum(image_test_resized, axis = 2)/765
	
	print image_test.shape, image_size.shape, image_size.shape, label_test.shape
	#print np.sum(image_gt), np.sum(image_test)

	#y_true = label_gt_resized.flatten()
	#y_pred = label_test.flatten()
	
	#y_true_all = label_gt.flatten()
	#y_pred_all = label_test_resized.flatten()
	
	#scores.append(jaccard_similarity_score(y_true, y_pred))
	#scores_original.append(jaccard_similarity_score(y_true_all, y_pred_all))

#print np.mean(scores), np.mean(scores_original)

	

	 

