import pickle
import os

fileDir = os.path.dirname(os.path.realpath(__file__))
feature_dir = os.path.join(fileDir, '..', 'web', 'captured', 'feature')
pklfile = os.path.join(feature_dir, 'classifier.pkl')

file = open(pklfile,"r")
le, clf = pickle.load(file)
print(le.classes_)

labels = ['kazhang', 'longyg', 'longyg']
labelsNum = le.transform(labels)
print(labelsNum)