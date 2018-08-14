# coding: utf-8
import numpy as np
import base64
import StringIO
from PIL import Image
import matplotlib.pyplot as plt
import urllib
import cv2
import sys
import os
import pandas as pd
import operator

fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import openface

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlib_model = os.path.join(modelDir, 'dlib', 'shape_predictor_68_face_landmarks.dat')
openface_model = os.path.join(modelDir, 'openface', 'nn4.small2.v1.t7')

# In[2]:

dataURL = "data:image/jpeg;base64,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"

head = "data:image/jpeg;base64,"
imgdata = base64.b64decode(dataURL[len(head):])
imgF = StringIO.StringIO()

# In[3]:
imgF.write(imgdata)
imgF.seek(0)

# In[4]:
img = Image.open(imgF)

# In[5]:
arr = np.asarray(img)

# In[6]:
buf = np.fliplr(arr)

# In[7]:
rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
rgbFrame[:, :, 0] = buf[:, :, 2]
rgbFrame[:, :, 1] = buf[:, :, 1]
rgbFrame[:, :, 2] = buf[:, :, 0]

# In[8]:
annotatedFrame = np.copy(buf)

# In[10]:
align = openface.AlignDlib(dlib_model)
net = openface.TorchNeuralNet(openface_model, imgDim=96, cuda=False)
bb = align.getLargestFaceBoundingBox(rgbFrame)

# In[12]:
bbs = [bb] if bb is not None else []

# In[17]:
for bb in bbs:
    # print(len(bbs))
    landmarks = align.findLandmarks(rgbFrame, bb)
    alignedFace = align.align(96, rgbFrame, bb,
                              landmarks=landmarks,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    rep = net.forward(alignedFace)
    print(rep)

feature_dir = os.path.join(fileDir, '..', 'web', 'captured', 'feature')
fname = os.path.join(feature_dir, 'labels.csv')
labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
print(labels)
map1 = map(os.path.dirname, labels)
print(map1)
map2 = map(os.path.split, map1)
print(map2)
map3 = map(operator.itemgetter(1), map2)
print(map3)
labels = map(operator.itemgetter(1), map(os.path.split, map(os.path.dirname, labels)))  # Get the directory.
print(labels)