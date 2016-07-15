#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/DPN2_Coco_val/DPN2_VOC_solver2.prototxt --weights=examples/DPN2_Coco_val/model/unary2_iter_60000.caffemodel --gpu 5 2>&1|tee examples/DPN2_Coco_val/finetune2.log
#./build/tools/caffe train --solver=examples/DPN_Coco_val/DPN_MS_VOC_solver.prototxt --weights=examples/DPN_Coco_val/model/finetune_iter_16000.caffemodel --gpu 0 2>&1|tee examples/DPN_Coco_val/finetune_ms.log