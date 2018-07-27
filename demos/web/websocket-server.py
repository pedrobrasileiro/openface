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
from twisted.internet.ssl import DefaultOpenSSLContextFactory

from twisted.python import log

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface
import subprocess
import predictor

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
capturedImageDir = os.path.join(fileDir, 'captured')
statisticFile = os.path.join(fileDir, 'captured', 'test', 'statistic.txt')

# For TLS connections
tls_crt = os.path.join(fileDir, 'tls', 'server.crt')
tls_key = os.path.join(fileDir, 'tls', 'server.key')

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
        self.saveImg(name, img)

    def processTestFrame(self, dataURL, name):
        img = self.getImg(dataURL)
        self.testImg(name, img)

    def processRecogFrame(self, dataURL):
        img = self.getImg(dataURL)
        self.recogImg(img)

    def saveImg(self, name, img):
        originDir, alignedDir = createCapturedImageDirs(name)
        number = getLatestImageNumber(originDir) + 1
        filename = str(number) + ".jpg"
        img.save(originDir + "/" + filename)

        filepath = "captured/origin/" + name + "/" + filename
        msg = {
            "type": "NEW_IMAGE",
            "name": name,
            "path": filepath
        }
        self.sendMessage(json.dumps(msg))

    def testImg(self, name, img):
        testdir = createTestImageDir(name)
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
        recogdir = createRecogImageDir()
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

    def train(self):
        print("Training is started...")
        command = "sh /root/ylong/workspace/openface/demos/web/train.sh"
        result = os.system(command)
        print("Training is completed with return code: ", result)
        if result == 0:
            status = "ok"
        else:
            status = "nok"
        msg = {
            "type": "TRAINED",
            "status": status
        }
        self.sendMessage(json.dumps(msg))

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

def createCapturedImageDirs(name):
    if not os.path.exists(capturedImageDir):
        os.mkdir(capturedImageDir)

    originDir = os.path.join(capturedImageDir, "origin")
    alignedDir = os.path.join(capturedImageDir, "aligned")
    if not os.path.exists(originDir):
        os.mkdir(originDir)
    if not os.path.exists(alignedDir):
        os.mkdir(alignedDir)

    originNameDir = os.path.join(originDir, name)
    alignedNameDir = os.path.join(alignedDir, name)
    if not os.path.exists(originNameDir):
        os.mkdir(originNameDir)
    if not os.path.exists(alignedNameDir):
        os.mkdir(alignedNameDir)

    return originNameDir, alignedNameDir

def createTestImageDir(name):
    if not os.path.exists(capturedImageDir):
        os.mkdir(capturedImageDir)
    testDir = os.path.join(capturedImageDir, "test")
    if not os.path.exists(testDir):
        os.mkdir(testDir)
    testNameDir = os.path.join(testDir, name)
    if not os.path.exists(testNameDir):
        os.mkdir(testNameDir)
    return testNameDir

def createRecogImageDir():
    if not os.path.exists(capturedImageDir):
        os.mkdir(capturedImageDir)
    recogDir = os.path.join(capturedImageDir, "recognition")
    if not os.path.exists(recogDir):
        os.mkdir(recogDir)
    return recogDir

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
    ctx_factory = DefaultOpenSSLContextFactory(tls_key, tls_crt)
    reactor.listenSSL(args.port, factory, ctx_factory)
    return defer.Deferred()

if __name__ == '__main__':
    task.react(main)
