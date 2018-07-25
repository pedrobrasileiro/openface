#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys
from sklearn.mixture import GMM
import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

imgDim = 96
align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(
          os.path.join(openfaceModelDir,'nn4.small2.v1.t7'),
          imgDim=imgDim,
          cuda=False)

def getRep(imgPath, multiple=False):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise IOException("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if multiple:
        bbs = align.getAllFaceBoundingBoxes(rgbImg)
    else:
        bb1 = align.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise NoFaceDetectedException("Unable to find a face: {}".format(imgPath))

    reps = []
    for bb in bbs:
        alignedFace = align.align(
            imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise UnableAlignException("Unable to align image: {}".format(imgPath))

        rep = net.forward(alignedFace)
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps

def infer(classifierModel, img):
    with open(classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
        else:
                (le, clf) = pickle.load(f, encoding='latin1')

    print("\n=== {} ===".format(img))
    reps = getRep(img, False)
    if len(reps) > 1:
        print("List of faces in image from left to right")
    for r in reps:
        rep = r[1].reshape(1, -1)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        print("Prediction took {} seconds.".format(time.time() - start))
        print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
    return person, confidence

class IOException(Exception):
    def __init__(self, arg):
        self.args = arg

class NoFaceDetectedException(Exception):
    def __init__(self, arg):
        self.args = arg

class UnableAlignException(Exception):
    def __init__(self, arg):
        self.args = arg
