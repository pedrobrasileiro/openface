#!/usr/bin/env python2
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

import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.internet import task, defer
# from twisted.internet.ssl import DefaultOpenSSLContextFactory
from OpenSSL import crypto
from twisted.internet import ssl

from twisted.python import log

import argparse
import json
from PIL import Image
import numpy as np
import os
import StringIO
import base64
import cv2
import matplotlib.pyplot as plt
import urllib

import matplotlib as mpl
mpl.use('Agg')

import openface
import predictor

import pickle

import pandas as pd
import operator
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from nolearn.dbn import DBN

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
capturedImageDir = os.path.join(fileDir, 'captured')
statisticFile = os.path.join(fileDir, 'captured', 'test', 'statistic.txt')
repDir = os.path.join(fileDir, 'representation')

# For TLS connections
# tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
# tls_key = os.path.join(fileDir, 'tls', 'server.key')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=args.cuda)


repDir = os.path.join(fileDir, 'representation')
if not os.path.exists(repDir):
    os.mkdir(repDir)

labels = np.array([])
labelsCsv = os.path.join(repDir, 'labels.csv')
if os.path.exists(labelsCsv):
    print("Loading '{}'".format(labelsCsv))
    tmp = pd.read_csv(labelsCsv, header=None).as_matrix()[:, 0]
    labels = list(tmp)

embeddings = np.array([])
repsCsv = os.path.join(repDir, 'reps.csv')
if os.path.exists(repsCsv):
    print("Loading '{}'".format(repsCsv))
    embeddings = pd.read_csv(repsCsv, header=None).as_matrix()


le = None
clf = None
classifierModel = os.path.join(repDir, 'classifier.pkl')
if os.path.exists(classifierModel):
    print("Loading '{}'".format(classifierModel))
    with open(classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')




class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):
    def __init__(self):
        super(OpenFaceServerProtocol, self).__init__()
        self.images = {}
        self.training = True
        self.people = []
        self.svm = None
        self.testing = False
        self.testedTotalImgs = 0
        self.successImgs = 0
        self.errorImgs = 0
        self.recognition = False
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)

        print("Received {} message of length {}.".format(msg['type'], len(raw)))

        # Face training
        if msg['type'] == "FRAME":
            self.processFrame(msg['dataURL'], msg['name'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAIN":
            self.train()
        elif msg['type'] == 'DELETE':
            self.deleteImage(msg['name'], msg['image'])

        # Face prediction test
        elif msg['type'] == "TEST":
            self.testing = True
            self.sendMessage('{"type": "TEST_STARTED"}')
        elif msg['type'] == "TEST_FRAME":
            self.processTestFrame(msg['dataURL'], msg['name'])
            self.sendMessage('{"type": "TEST_PROCESSED"}')
        elif msg['type'] == "STOP_TEST":
            self.testing = False
            self.sendMessage('{"type": "TEST_STOPPED"}')
        elif msg['type'] == "STAT":
            self.statistic()

        # Face recognition
        elif msg['type'] == "RECOGNITION":
            self.recognition = True
            self.sendMessage('{"type": "RECOGNITION_STARTED"}')
        elif msg['type'] == "RECOGNITION_FRAME":
            self.processRecogFrame(msg['dataURL'])
            self.sendMessage('{"type": "RECOGNITION_PROCESSED"}')
        elif msg['type'] == "STOP_RECOGNITION":
            self.recognition = False
            self.sendMessage('{"type": "RECOGNITION_STOPPED"}')
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def deleteImage(self, name, image):
        originDir = getOriginDir(name)
        filename = os.path.basename(image)
        filepath = os.path.join(originDir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)

    def getImg(self, dataURL):
        head = "data:image/jpeg;base64,"
        assert (dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)
        return img

    def processFrame(self, dataURL, name):
        img = self.getImg(dataURL)
        self.saveImage(name, img)

    def processTestFrame(self, dataURL, name):
        img = self.getImg(dataURL)
        self.testImg(name, img)

    def processRecogFrame(self, dataURL):
        img = self.getImg(dataURL)
        self.recogImg(img)

    def saveImageToFile(self, name, img):
        originDir = getOriginDir(name)
        number = getLatestImageNumber(originDir) + 1
        filename = str(number) + ".jpg"
        img.save(originDir + "/" + filename)

    def saveImage(self, name, img):
        self.saveImageToFile(name, img)

        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        annotatedFrame = np.copy(buf)

        bb = align.getLargestFaceBoundingBox(rgbFrame)
        if bb is None:
            return

        landmarks = align.findLandmarks(rgbFrame, bb)
        alignedFace = align.align(args.imgDim, rgbFrame, bb,
                                  landmarks=landmarks,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            return

        rep = net.forward(alignedFace)

        global labels
        tmp = np.append(labels, name)
        labels = list(tmp)

        global embeddings
        tmp = np.append(embeddings, [rep], axis=0)
        embeddings = tmp

        self.saveRepToFile(rep, name)

        bl = (bb.left(), bb.bottom())
        tr = (bb.right(), bb.top())
        cv2.rectangle(annotatedFrame, bl, tr, color=(255, 255, 255),
                      thickness=1)
        self.sendImage(annotatedFrame, name)

    def sendImage(self, frame, name):
        plt.figure()
        plt.imshow(frame)
        plt.xticks([])
        plt.yticks([])

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))

        msg = {
            "type": "NEW_ALIGNED_IMAGE",
            "content": content,
            "name": name
        }
        plt.close()
        self.sendMessage(json.dumps(msg))

    def sendAlignedImage(self, alignedFace, name):
        content = [str(x) for x in alignedFace.flatten()]
        msg = {
            "type": "NEW_ALIGNED_IMAGE",
            "content": content,
            "name": name
        }
        self.sendMessage(json.dumps(msg))

    def saveRepToFile(self, rep, name):
        repDir = getRepDir()
        labelFile = os.path.join(repDir, 'labels.csv')
        repFile = os.path.join(repDir, 'reps.csv')

        lf = open(labelFile, 'a')
        lf.write(name + '\n')

        rf = open(repFile, 'a')
        repstrlist = [str(r) for r in rep]
        rf.write(','.join(repstrlist) + '\n')

        lf.close()
        rf.close()

    def testImg(self, name, img):
        testdir = getTestDir(name)
        number = getLatestImageNumber(testdir) + 1
        filename = str(number) + ".jpg"
        filepath = testdir + "/" + filename
        img.save(filepath)

        imgpath = "captured/test/" + name + "/" + filename
        try:
            predict, confidence = predictor.infer(os.path.join(capturedImageDir, "feature", "classifier.pkl"), filepath)

            if name == predict:
                predict_result = "predict is correct"
            else:
                predict_result = "predict is wrong, predict: " + predict + ", actual: " + name

            self.writetofile(name, filepath, predict, confidence, predict_result)
            msg = {
                "type": "NEW_TEST_IMAGE",
                "actual_name": name,
                "predict_name": predict,
                "predict_result": predict_result,
                "confidence": confidence,
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))
        except predictor.IOException:
            self.writetofile(name, filepath, "unknown", "unknown", "error: Unable to load image")
            msg = {
                "type": "NEW_TEST_IMAGE",
                "actual_name": name,
                "predict_name": "unknown",
                "predict_result": "error: Unable to load image",
                "confidence": "unknown",
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))
        except predictor.NoFaceDetectedException:
            self.writetofile(name, filepath, "unknown", "unknown", "error: No face detected")
            msg = {
                "type": "NEW_TEST_IMAGE",
                "actual_name": name,
                "predict_name": "unknown",
                "predict_result": "error: No face detected",
                "confidence": "unknown",
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))
        except predictor.UnableAlignException:
            self.writetofile(name, filepath, "unknown", "unknown", "error: Unable to align image")
            msg = {
                "type": "NEW_TEST_IMAGE",
                "actual_name": name,
                "predict_name": "unknown",
                "predict_result": "error: Unable to align image",
                "confidence": "unknown",
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))

    def recogImg(self, img):
        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        try:
            predict, confidence = self.infer(rgbFrame)
            self.sendRecogStatus(predict, confidence, "success")
        except Exception, e:
            self.sendRecogStatus("unknown", "unknown", e.message)

    def sendRecogStatus(self, predict, confidence, result):
        msg = {
            "type": "NEW_RECOGNITION_IMAGE",
            "predict_name": predict,
            "confidence": confidence,
            "predict_result": result
        }
        self.sendMessage(json.dumps(msg))

    def recogImg0(self, img):
        recogdir = getRecogDir()
        number = getLatestImageNumber(recogdir) + 1
        filename = str(number) + ".jpg"
        filepath = recogdir + "/" + filename
        img.save(filepath)
        imgpath = "captured/recognition/" + filename

        try:
            predict, confidence = predictor.infer(os.path.join(capturedImageDir, "feature", "classifier.pkl"), filepath)
            msg = {
                "type": "NEW_RECOGNITION_IMAGE",
                "predict_name": predict,
                "confidence": confidence,
                "predict_result": "success",
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))
        except predictor.IOException:
            msg = {
                "type": "NEW_RECOGNITION_IMAGE",
                "predict_name": "unknown",
                "confidence": "unknown",
                "predict_result": "IO exception",
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))
        except predictor.NoFaceDetectedException:
            msg = {
                "type": "NEW_RECOGNITION_IMAGE",
                "predict_name": "unknown",
                "confidence": "unknown",
                "predict_result": "No face detected",
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))
        except predictor.UnableAlignException:
            msg = {
                "type": "NEW_RECOGNITION_IMAGE",
                "predict_name": "unknown",
                "confidence": "unknown",
                "predict_result": "Unable to align image",
                "image": imgpath
            }
            self.sendMessage(json.dumps(msg))

    def writetofile(self, name, filepath, predict_name, confidence, predict_result):
        with open(statisticFile, 'a') as f:
            line = ','.join((name, filepath, predict_name, str(confidence), predict_result))
            f.write(line + "\n")

    def train(self, classifier='LinearSvm', ldaDim=1):
        # print("Loading embeddings.")
        # fname = "{}/labels.csv".format(getRepDir())
        # labels = pd.read_csv(fname, header=None).as_matrix()[:, 0]
        # labels = map(operator.itemgetter(1),
        #              map(os.path.split,
        #                  map(os.path.dirname, labels)))  # Get the directory.
        # fname = "{}/reps.csv".format(getRepDir())
        # embeddings = pd.read_csv(fname, header=None).as_matrix()

        global le
        global clf

        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))

        if classifier == 'LinearSvm':
            clf = SVC(C=1, kernel='linear', probability=True)
        elif classifier == 'GridSearchSvm':
            print("""
            Warning: In our experiences, using a grid search over SVM hyper-parameters only
            gives marginally better performance than a linear SVM with C=1 and
            is not worth the extra computations of performing a grid search.
            """)
            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif classifier == 'GMM':  # Doesn't work best
            clf = GMM(n_components=nClasses)

        # ref:
        # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
        elif classifier == 'RadialSvm':  # Radial Basis Function kernel
            # works better with C = 1 and gamma = 2
            clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif classifier == 'DecisionTree':  # Doesn't work best
            clf = DecisionTreeClassifier(max_depth=20)
        elif classifier == 'GaussianNB':
            clf = GaussianNB()

        # ref: https://jessesw.com/Deep-Learning/
        elif classifier == 'DBN':
            clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                      learn_rates=0.3,
                      # Smaller steps mean a possibly more accurate result, but the
                      # training will take longer
                      learn_rate_decays=0.9,
                      # a factor the initial learning rate will be multiplied by
                      # after each iteration of the training
                      epochs=300,  # no of iternation
                      # dropouts = 0.25, # Express the percentage of nodes that
                      # will be randomly dropped as a decimal.
                      verbose=1)

        if ldaDim > 0:
            clf_final = clf
            clf = Pipeline([('lda', LDA(n_components=ldaDim)),
                            ('clf', clf_final)])

        clf.fit(embeddings, labelsNum)

        self.saveModel(le, clf)

        self.sendTrainStatus("ok")

    def saveModel(self, le, clf):
        fName = "{}/classifier.pkl".format(getRepDir())
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'w') as f:
            pickle.dump((le, clf), f)

    def sendTrainStatus(self, status):
        msg = {
            "type": "TRAINED",
            "status": status
        }
        self.sendMessage(json.dumps(msg))

    def getRep(self, rgbImg, multiple=False):
        if multiple:
            bbs = align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
        if len(bbs) == 0 or (not multiple and bb1 is None):
            raise NoFaceDetectedException("Unable to find a face")

        reps = []
        for bb in bbs:
            alignedFace = align.align(
                args.imgDim,
                rgbImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                raise UnableAlignException("Unable to align face")

            rep = net.forward(alignedFace)
            reps.append((bb.center().x, rep))
        sreps = sorted(reps, key=lambda x: x[0])
        return sreps

    def infer(self, rgbImg):
        reps = self.getRep(rgbImg)
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for r in reps:
            rep = r[1].reshape(1, -1)
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
            if isinstance(clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))
        return person, confidence

    def statistic(self):
        statResult = {}

        with open(statisticFile, 'r') as f:
            for line in f:
                strings = line.split(',')
                name = strings[0].strip()
                predict_name = strings[2].strip()
                if statResult.get(name) is None:
                    total = 0
                    success = 0
                    error = 0
                else:
                    total = statResult.get(name).get("total")
                    success = statResult.get(name).get("success")
                    error = statResult.get(name).get("error")
                if predict_name != "unknown":
                    total += 1
                    if predict_name == name:
                        success += 1
                    else:
                        error += 1
                statResult[name] = {"name": name, "total": total, "success": success, "error": error}
        msg = {
            "type": "STATED",
            "statResult": statResult
        }
        self.sendMessage(json.dumps(msg))

class IOException(Exception):
    def __init__(self, arg):
        self.args = arg

class NoFaceDetectedException(Exception):
    def __init__(self, arg):
        self.args = arg

class UnableAlignException(Exception):
    def __init__(self, arg):
        self.args = arg

def getOriginDir(name):
    if not os.path.exists(capturedImageDir):
        os.mkdir(capturedImageDir)

    originDir = os.path.join(capturedImageDir, "origin")
    if not os.path.exists(originDir):
        os.mkdir(originDir)

    originNameDir = os.path.join(originDir, name)
    if not os.path.exists(originNameDir):
        os.mkdir(originNameDir)

    return originNameDir

def getAlignedDir(name):
    if not os.path.exists(capturedImageDir):
        os.mkdir(capturedImageDir)

    alignedDir = os.path.join(capturedImageDir, "aligned")
    if not os.path.exists(alignedDir):
        os.mkdir(alignedDir)

    alignedNameDir = os.path.join(alignedDir, name)
    if not os.path.exists(alignedNameDir):
        os.mkdir(alignedNameDir)

    return alignedNameDir

def getTestDir(name):
    if not os.path.exists(capturedImageDir):
        os.mkdir(capturedImageDir)
    testDir = os.path.join(capturedImageDir, "test")
    if not os.path.exists(testDir):
        os.mkdir(testDir)
    testNameDir = os.path.join(testDir, name)
    if not os.path.exists(testNameDir):
        os.mkdir(testNameDir)
    return testNameDir

def getRecogDir():
    if not os.path.exists(capturedImageDir):
        os.mkdir(capturedImageDir)
    recogDir = os.path.join(capturedImageDir, "recognition")
    if not os.path.exists(recogDir):
        os.mkdir(recogDir)
    return recogDir

def getRepDir():
    if not os.path.exists(repDir):
        os.mkdir(repDir)
    return repDir

def getLatestImageNumber(dir):
    numbers = []
    for path, dirs, files in os.walk(dir):
        for file in files:
            filename = os.path.splitext(file)[0]
            numbers.append(int(filename))
    if len(numbers) < 1:
        return 0
    else:
        return max(numbers)

def main(reactor):
    log.startLogging(sys.stdout)
    factory = WebSocketServerFactory()
    factory.protocol = OpenFaceServerProtocol
    # ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)

    privkey = os.path.join(fileDir, '..', '..', 'tls', 'privkey.pem')
    cert = os.path.join(fileDir, '..', '..', 'tls', 'cert.pem')
    chain = os.path.join(fileDir, '..', '..', 'tls', 'chain.pem')

    privkey=open(privkey, 'rt').read()
    certif=open(cert, 'rt').read()
    chain=open(chain, 'rt').read()

    privkeypyssl=crypto.load_privatekey(crypto.FILETYPE_PEM,privkey)
    certifpyssl=crypto.load_certificate(crypto.FILETYPE_PEM,certif)
    chainpyssl=[crypto.load_certificate(crypto.FILETYPE_PEM,chain)]
    ctx_factory = ssl.CertificateOptions(privateKey=privkeypyssl,certificate=certifpyssl,extraCertChain=chainpyssl)

    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
