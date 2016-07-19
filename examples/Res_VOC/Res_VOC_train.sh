#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/Res_VOC/Res_VOC_solver.prototxt --weights=examples/Res_VOC/ResNet-101-model.caffemodel --gpu 1 2>&1|tee examples/Res_VOC/train.log
#./build/tools/caffe train --solver=examples/DPN_Coco_val/DPN_MS_VOC_solver.prototxt --weights=examples/DPN_Coco_val/model/finetune_iter_16000.caffemodel --gpu 0 2>&1|tee examples/DPN_Coco_val/finetune_ms.log