% This script contains pipeline of FCN + DenseCRF
clear; clc;
if exist('../../+caffe', 'dir')
  addpath('../../');
end;
caffe.reset_all();

%%
use_gpu = 0;

if use_gpu ~= -1
    caffe.set_mode_gpu();
    caffe.set_device(use_gpu);
else
    caffe.set_mode_cpu();
end
caffe_solver = caffe.Solver('feature_selection_test_solver.prototxt');

a = rand(500, 1, 256);
b = randi(500, 4000, 3) - 1;
c = rand(4000, 1, 3, 256);

input_data = {single(permute(a, [2, 1, 3])), single(permute(b, [2, 1, 3])), single(permute(c, [3, 2, 4, 1]))};


  caffe_solver.net.set_phase('train');
  caffe_solver.net.reshape_as_input(input_data);
  caffe_solver.net.set_input_data(input_data);
  caffe_solver.step(1);



