% This script contains pipeline of FCN + DenseCRF
clear; clc;
caffe.reset_all();

caffe.set_mode_cpu();
net = caffe.Net('deploy.prototxt', 'VGG_16.caffemodel', 'train');

%%

label = randi(2, 0, 3, 3, 'single')-1;
predict = randi(2, 5, 3, 3, 'single')-1;
score = rand(5, 1, 1, 'single');



input = {permute(label, [3, 2, 1]); permute(predict, [3, 2, 1]); permute(score, [3, 2, 1])};

score = net.forward(input);