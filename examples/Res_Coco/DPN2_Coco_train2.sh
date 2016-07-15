#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/DPN2_Coco_val/DPN2_Coco_solver2.prototxt --weights=examples/DPN2_Coco_val/unary3_iter_9000.caffemodel --gpu 5 2>&1|tee examples/DPN2_Coco_val/train2.log