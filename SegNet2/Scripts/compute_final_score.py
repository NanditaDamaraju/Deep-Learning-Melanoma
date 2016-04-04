import json, glob
from sklearn.metrics import average_precision_score
import numpy as np
score_globules = []
score_streaks  = []
i=0
for name in sorted(glob.glob('/home/ubuntu/data/SegNet2/Fextract/predicted_superpixels/ISIC_00?????.json')):
	with open(name) as data_file:
		data = json.load(data_file)
	name2 = '/home/ubuntu/data/SegNet2/Fextract/ISBI/TrGT/'+name[57:]
	with open(name2) as data_file:
		data2 = json.load(data_file)
	scoreg = average_precision_score(data["globules"], data2["globules"])  
	scores = average_precision_score(data["streaks"], data2["streaks"])
	print scoreg
	print scores
	score_globules.append(scoreg)
	score_streaks.append(scores)

print "Mean Scores"
print np.array(scoreg).mean()
print np.array(scores).mean()
