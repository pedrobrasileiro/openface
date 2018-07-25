import pickle

with open('./captured/feature/classifier.pkl', 'rb') as f:
    le, clf = pickle.load(f)
f.close()
print(le)
print(clf)
