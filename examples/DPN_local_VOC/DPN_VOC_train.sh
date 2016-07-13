#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/DPN_VOC/DPN_VOC_solver.prototxt --weights=examples/DPN_VOC/VGG_16.caffemodel --gpu 0 2>&1|tee examples/DPN_VOC/train.log