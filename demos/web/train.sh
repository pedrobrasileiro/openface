#!/usr/bin/bash

raw_image_dir=/root/ylong/workspace/openface/demos/web/captured/origin
aligned_image_dir=/root/ylong/workspace/openface/demos/web/captured/aligned
feature_dir=/root/ylong/workspace/openface/demos/web/captured/feature

for N in {1..8}
do
    /root/ylong/workspace/openface/util/align-dlib.py $raw_image_dir align outerEyesAndNose $aligned_image_dir --size 96
done

/root/ylong/workspace/openface/batch-represent/main.lua -outDir $feature_dir -data $aligned_image_dir

/root/ylong/workspace/openface/demos/classifier.py train $feature_dir