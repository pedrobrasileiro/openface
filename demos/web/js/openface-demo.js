/*
Copyright 2015-2016 Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) ?
        function(c, os, oe) {
            navigator.mediaDevices.getUserMedia(c).then(os,oe);
        } : null ||
    navigator.msGetUserMedia;

window.URL = window.URL ||
    window.webkitURL ||
    window.msURL ||
    window.mozURL;

// http://stackoverflow.com/questions/6524288
$.fn.pressEnter = function(fn) {

    return this.each(function() {
        $(this).bind('enterPress', fn);
        $(this).keyup(function(e){
            if(e.keyCode == 13)
            {
              $(this).trigger("enterPress");
            }
        })
    });
 };

function registerHbarsHelpers() {
    // http://stackoverflow.com/questions/8853396
    Handlebars.registerHelper('ifEq', function(v1, v2, options) {
        if(v1 === v2) {
            return options.fn(this);
        }
        return options.inverse(this);
    });
}

function startCapture() {
    var name = $("#name").val();
    if (name == "") {
        alert("Please input your name");
        return;
    }

    if (navigator.getUserMedia) {
         var videoSelector = {video : true};
         navigator.getUserMedia(videoSelector, videoSucess, function() {
             alert("Error fetching video from webcam");
         });
     } else {
         alert("No webcam detected.");
     }
}

function stopCapture() {
    vid.pause();

    if (vid.mozCaptureStream) {
        vid.mozSrcObject = null;
    } else {
        vid.src = "" || null;
    }

    if (localStream) {
        if (localStream.getVideoTracks) {
            // get video track to call stop on it
            var tracks = localStream.getVideoTracks();
            if (tracks && tracks[0] && tracks[0].stop) tracks[0].stop();
        }
        else if (localStream.stop) {
            // deprecated, may be removed in future
            localStream.stop();
        }
        localStream = null;
    }

    vidReady = false;
}

function videoSucess(stream) {
    if (vid.mozCaptureStream) {
        vid.mozSrcObject = stream;
    } else {
        vid.src = (window.URL && window.URL.createObjectURL(stream)) ||
            stream;
    }
    localStream = stream;
    vid.play();
    vidReady = true;
    setTimeout(function() {requestAnimFrame(sendFrameLoop)}, 1000);
}


function sendFrameLoop() {
    if (socket == null || socket.readyState != socket.OPEN ||
        !vidReady) {
        return;
    }

    if (tok > 0) {
        console.log("starting to send frame message")
        var canvas = document.createElement('canvas');
        canvas.width = vid.width;
        canvas.height = vid.height;
        var cc = canvas.getContext('2d');
        cc.drawImage(vid, 0, 0, vid.width, vid.height);
        var apx = cc.getImageData(0, 0, vid.width, vid.height);

        var dataURL = canvas.toDataURL('image/jpeg', 0.6)

        var msg = {
            'type': 'FRAME',
            'dataURL': dataURL,
            'name': $("#name").val()
        };
        socket.send(JSON.stringify(msg));
        tok--;
    }
    setTimeout(function() {requestAnimFrame(sendFrameLoop)}, 500);
}

function startTrain() {
    var msg = {
        'type': 'TRAIN'
    };
    socket.send(JSON.stringify(msg));
    $("#trainStatus").html("Training is ongoing...");
}

function redrawPeople() {
    var context = {images: images};
    $("#peopleTd").html(peopleTableTmpl(context));
}

//////////////////////// Face prediction test
function startTest() {
    var name = $("#testName").val();
    if (name == "") {
        alert("Please input your name");
        return;
    }

    $("#testStatus").html("Test is starting...");

    var msg = {
        'type': 'TEST'
    };
    socket.send(JSON.stringify(msg));
}

function stopTest() {
    $("#testStatus").html("Test is stopping...");
    stopTestCapture();
    var msg = {
        'type': 'STOP_TEST'
    };
    socket.send(JSON.stringify(msg));
}

function startTestCapture() {
    $("#testStatus").html("Test is ongoing...");
    if (navigator.getUserMedia) {
         var videoSelector = {video : true};
         navigator.getUserMedia(videoSelector, testVideoSucess, function() {
             alert("Error fetching video from webcam");
         });
     } else {
         alert("No webcam detected.");
     }
}

function stopTestCapture() {
    testVid.pause();

    if (testVid.mozCaptureStream) {
        testVid.mozSrcObject = null;
    } else {
        testVid.src = "" || null;
    }

    if (testLocalStream) {
        if (testLocalStream.getVideoTracks) {
            // get video track to call stop on it
            var tracks = testLocalStream.getVideoTracks();
            if (tracks && tracks[0] && tracks[0].stop) tracks[0].stop();
        }
        else if (testLocalStream.stop) {
            // deprecated, may be removed in future
            testLocalStream.stop();
        }
        testLocalStream = null;
    }

    testVidReady = false;
}

function testVideoSucess(stream) {
    if (testVid.mozCaptureStream) {
        testVid.mozSrcObject = stream;
    } else {
        testVid.src = (window.URL && window.URL.createObjectURL(stream)) ||
            stream;
    }
    testLocalStream = stream;
    testVid.play();
    testVidReady = true;
    setTimeout(function() {requestAnimFrame(sendTestFrameLoop)}, 1000);
}

function sendTestFrameLoop() {
    if (socket == null || socket.readyState != socket.OPEN ||
        !testVidReady) {
        return;
    }

    if (testTok > 0) {
        var canvas = document.createElement('canvas');
        canvas.width = testVid.width;
        canvas.height = testVid.height;
        var cc = canvas.getContext('2d');
        cc.drawImage(testVid, 0, 0, testVid.width, testVid.height);
        var apx = cc.getImageData(0, 0, testVid.width, testVid.height);

        var dataURL = canvas.toDataURL('image/jpeg', 0.6)

        var msg = {
            'type': 'TEST_FRAME',
            'dataURL': dataURL,
            'name': $("#testName").val()
        };
        socket.send(JSON.stringify(msg));
        testTok--;
    }
    setTimeout(function() {requestAnimFrame(sendTestFrameLoop)}, 500);
}

function startStat() {
    $("#statistic").html("statistic is ongoing...");
    var msg = {
        'type': 'STAT'
    };
    socket.send(JSON.stringify(msg));
}

////////////// Face Recognition
function startRecognition() {
    $("#recogStatus").html("Face recognition is starting...");

    var msg = {
        'type': 'RECOGNITION'
    };
    socket.send(JSON.stringify(msg));
}

function stopRecognition() {
    $("#recogStatus").html("Face recognition is stopping...");
    stopRecogCapture();
    var msg = {
        'type': 'STOP_RECOGNITION'
    };
    socket.send(JSON.stringify(msg));
}

function startRecogCapture() {
    $("#recogStatus").html("Face recognition is ongoing...");
    if (navigator.getUserMedia) {
         var videoSelector = {video : true};
         navigator.getUserMedia(videoSelector, recogVideoSucess, function() {
             alert("Error fetching video from webcam");
         });
     } else {
         alert("No webcam detected.");
     }
}

function stopRecogCapture() {
    recogVid.pause();

    if (recogVid.mozCaptureStream) {
        recogVid.mozSrcObject = null;
    } else {
        recogVid.src = "" || null;
    }

    if (recogLocalStream) {
        if (recogLocalStream.getVideoTracks) {
            // get video track to call stop on it
            var tracks = recogLocalStream.getVideoTracks();
            if (tracks && tracks[0] && tracks[0].stop) tracks[0].stop();
        }
        else if (recogLocalStream.stop) {
            // deprecated, may be removed in future
            recogLocalStream.stop();
        }
        recogLocalStream = null;
    }

    recogVidReady = false;
}

function recogVideoSucess(stream) {
    if (recogVid.mozCaptureStream) {
        recogVid.mozSrcObject = stream;
    } else {
        recogVid.src = (window.URL && window.URL.createObjectURL(stream)) ||
            stream;
    }
    recogLocalStream = stream;
    recogVid.play();
    recogVidReady = true;
    setTimeout(function() {requestAnimFrame(sendRecogFrameLoop)}, 1000);
}

function sendRecogFrameLoop() {
    if (socket == null || socket.readyState != socket.OPEN ||
        !recogVidReady) {
        return;
    }

    if (recogTok > 0) {
        var canvas = document.createElement('canvas');
        canvas.width = recogVid.width;
        canvas.height = recogVid.height;
        var cc = canvas.getContext('2d');
        cc.drawImage(recogVid, 0, 0, recogVid.width, recogVid.height);
        var apx = cc.getImageData(0, 0, recogVid.width, recogVid.height);

        var dataURL = canvas.toDataURL('image/jpeg', 0.6)

        var msg = {
            'type': 'RECOGNITION_FRAME',
            'dataURL': dataURL
        };
        socket.send(JSON.stringify(msg));
        recogTok--;
    }
    setTimeout(function() {requestAnimFrame(sendRecogFrameLoop)}, 500);
}

///////////////// Create web socket
function createSocket(address, name) {
    socket = new WebSocket(address);
    socketName = name;
    socket.binaryType = "arraybuffer";
    socket.onopen = function() {
        $("#serverStatus").html("Connected to " + name);
    };
    socket.onmessage = function(e) {
        console.log(e);
        j = JSON.parse(e.data);
        if (j.type == "TRAINED") {
            if (j.status == "ok") {
                $("#trainStatus").html("<span style='color: green'>Training is done successfully!</span>");
            } else {
                $("#trainStatus").html("<span style='color: red'>Training is failed!</span>");
            }
        } else if (j.type == "PROCESSED") {
            tok++;
        } else if (j.type == "NEW_IMAGE") {
            images.push({
                name: j.name,
                image: j.path
            });
            redrawPeople();
        } else if (j.type == "TEST_STARTED") {
            startTestCapture();
        } else if (j.type == "TEST_PROCESSED") {
            testTok++;
        } else if (j.type == "TEST_STOPPED") {
            $("#testStatus").html("Test is stopped.");
        } else if (j.type == "NEW_TEST_IMAGE") {

            var outHtml = '<h4>Test Result:</h4>' +
                '<span style="font-weight: bold">Actual Name: </span><span>' + j.actual_name + '</span><br>';
            if (j.actual_name == j.predict_name) {
                outHtml += '<span style="font-weight: bold">Predict Name: </span><span style="color:green">' + j.predict_name + '</span><br>';
            } else {
                outHtml += '<span style="font-weight: bold">Predict Name: </span><span style="color:red">' + j.predict_name + '</span><br>';
            }
            if (j.confidence == "unknown") {
                outHtml += '<span style="font-weight: bold">Predict Confidence: </span><span style="color:red">' + j.confidence + '</span><br>';
            } else {
                outHtml += '<span style="font-weight: bold">Predict Confidence: </span><span>' + j.confidence + '</span><br>';
            }
            if (j.predict_result == "predict is correct") {
                outHtml += '<span style="font-weight: bold">Predict Result: </span><span style="color:green">' + j.predict_result + '</span>';
            } else {
                outHtml += '<span style="font-weight: bold">Predict Result: </span><span style="color:red">' + j.predict_result + '</span>';
            }

            $("#testResult").html(outHtml);

        } else if (j.type == "STATED") {
            $("#statStatus").html("<span style='color:green'>Statistic is done successfully!</span>");

            var result = j.statResult;
            var outHtml = "";
            for (var key in result) {
                outHtml += "<tr>" +
                                "<td>" + key + "</td>" +
                                "<td>" + result[key].total + "</td>" +
                                "<td>" + result[key].success + "</td>" +
                                "<td>" + result[key].error + "</td>" +
                                "<td>" + (result[key].success / result[key].total) + "</td>" +
                            "</tr>"
            }
            $("#statBody").html(outHtml);
        }

        else if (j.type == "RECOGNITION_STARTED") {
            startRecogCapture();
        } else if (j.type == "RECOGNITION_PROCESSED") {
            recogTok++;
        } else if (j.type == "RECOGNITION_STOPPED") {
            $("#testStatus").html("Face recognition is stopped.");
        } else if (j.type == "NEW_RECOGNITION_IMAGE") {
            var outHtml = '<h4>Face Recognition Result:</h4>';
            if (j.predict_name != "unknown") {
                outHtml += '<span style="font-weight: bold">People In Video: </span><span style="color:green">' + j.predict_name + '</span><br>';
            } else {
                outHtml += '<span style="font-weight: bold">People In Video: </span><span style="color:red">' + j.predict_name + '</span><br>';
            }

            $("#recogResult").html(outHtml);
        }

        else {
            console.log("Unrecognized message type: " + j.type);
        }
    }
    socket.onerror = function(e) {
        console.log("Error creating WebSocket connection to " + address);
        console.log(e);
    }
    socket.onclose = function(e) {
        if (e.target == socket) {
            $("#serverStatus").html("Disconnected.");
        }
    }
}
