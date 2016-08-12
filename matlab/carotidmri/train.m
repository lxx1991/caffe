clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 1;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);
%  caffe.set_mode_cpu();

%%
dir_model = fullfile('..', '..', 'examples', 'carotidmri');
file_solver = fullfile(dir_model, 'Res_Mat_solver.prototxt');
file_weight = fullfile(dir_model, 'ResNet-101-model.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);

temp = caffe_solver.net.layers('conv1').params.get_data();
caffe_solver.net.layers('conv1_2').params.set_data(temp);
caffe_solver.net.layers('conv1_3').params.set_data(temp);
caffe_solver.net.layers('conv1_4').params.set_data(temp);

dir_dataset = fullfile('..', '..', 'data', 'carotidmri');
train_list = fullfile(dir_dataset, 'train_list');
dir_dataset = fullfile(dir_dataset, 'images');

[name_list1, name_list2, name_list3, name_list4, label_list]= textread(train_list, '%s %s %s %s %s');

idx = randperm(length(label_list));
name_list1 = name_list1(idx);
name_list2 = name_list2(idx);
name_list3 = name_list3(idx);
name_list4 = name_list4(idx);
label_list = label_list(idx);

IMAGE_DIM = 320;
d = load('vgg16_mean');
IMAGE_MEAN = mean(d.image_mean);

idx = 1; show = 100;
%%
while (caffe_solver.iter() < caffe_solver.max_iter())
    
    if (~exist(fullfile(dir_dataset, [name_list1{idx}]), 'file'))
        idx = mod(idx,length(name_list1)) + 1;
        continue;
    end;
    
    img1 = repmat(imread(fullfile(dir_dataset, name_list1{idx})), 1, 1, 3);
    img2 = repmat(imread(fullfile(dir_dataset, name_list2{idx})), 1, 1, 3);
    img3 = repmat(imread(fullfile(dir_dataset, name_list3{idx})), 1, 1, 3);
    img4 = repmat(imread(fullfile(dir_dataset, name_list4{idx})), 1, 1, 3);
    label = imread(fullfile(dir_dataset, label_list{idx}));
    
    [ img1, img2, img3, img4, label ] = im_tf( img1, img2, img3, img4, label);
    
    input_img1 = single(img1(:, :, :)) - IMAGE_MEAN;
    input_img2 = single(img2(:, :, :)) - IMAGE_MEAN;
    input_img3 = single(img3(:, :, :)) - IMAGE_MEAN;
    input_img4 = single(img4(:, :, :)) - IMAGE_MEAN;
    
    input_img1 = permute(single(input_img1), [2, 1, 3]);
    input_img2 = permute(single(input_img2), [2, 1, 3]);
    input_img3 = permute(single(input_img3), [2, 1, 3]);
    input_img4 = permute(single(input_img4), [2, 1, 3]);
    input_label = permute(single(label), [2, 1, 3]);
    
    net_inputs = {input_img1, input_img2, input_img3, input_img4, input_label};
    
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    tic;
    caffe_solver.step_sample();
    toc;
    if (exist('show', 'var') && mod(caffe_solver.iter(), show) == 0)
        subplot(321)
        imshow(img1)
        subplot(322)
        imshow(img2)
        subplot(323)
        imshow(img3)
        subplot(324)
        imshow(img4)
        subplot(325)
        cmap = VOClabelcolormap(7);
        imshow(label, cmap);
        score = caffe_solver.net.blobs('upscore').get_data();
        [~, result] = max(permute(score, [2, 1, 3]), [], 3);
        subplot(326);
        imshow(result, cmap);
        drawnow;
    end;
    idx = mod(idx,length(name_list1)) + 1;
end;
