import csv, glob
import cPickle as pickle

f2 = open("/home/ubuntu/data/Phase3/Parsed/TestLabel.txt", "rb")
proba = pickle.load(f2)
f2.close()

i=0
with open('FinalResult.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for name in sorted(glob.glob('/home/ubuntu/data/Phase3/Test/Data/ISIC_00?????.jpg')):
	spamwriter.writerow([name[35:], proba[i,1]])
	i=i+1;



#    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
