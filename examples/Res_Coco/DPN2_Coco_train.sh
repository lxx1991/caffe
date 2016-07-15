#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/DPN2_Coco_val/DPN2_Coco_solver.prototxt --weights=examples/DPN2_Coco_val/ResNet-101-model.caffemodel --gpu 4 2>&1|tee examples/DPN2_Coco_val/train.log