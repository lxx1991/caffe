clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 0;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);
%  caffe.set_mode_cpu();

%%
dir_model = fullfile('..', '..', 'examples', 'carotidmri');
file_def_model = fullfile(dir_model, 'Res_Mat_test.prototxt');
file_model = fullfile(dir_model, 'model', 'unary_mat_iter_9000.caffemodel');

net = caffe.Net(file_def_model, file_model, 'test');

dir_dataset = fullfile('..', '..', 'data', 'carotidmri');
train_list = fullfile(dir_dataset, 'valid_list');
dir_dataset = fullfile(dir_dataset, 'images');

[name_list1, name_list2, name_list3, name_list4, label_list]= textread(train_list, '%s %s %s %s %s');

IMAGE_DIM = 320;
d = load('vgg16_mean');
IMAGE_MEAN = mean(d.image_mean);

%show = 100;
%%
for idx = 1:length(name_list1)
    
    if (~exist(fullfile(dir_dataset, [name_list1{idx}]), 'file'))
        continue;
    end;
    
    img1 = repmat(imread(fullfile(dir_dataset, name_list1{idx})), 1, 1, 3);
    img2 = repmat(imread(fullfile(dir_dataset, name_list2{idx})), 1, 1, 3);
    img3 = repmat(imread(fullfile(dir_dataset, name_list3{idx})), 1, 1, 3);
    img4 = repmat(imread(fullfile(dir_dataset, name_list4{idx})), 1, 1, 3);
    label = imread(fullfile(dir_dataset, label_list{idx}));
    
    [ input_img1, input_img2, input_img3, input_img4, ~ ] = im_tf2( img1, img2, img3, img4, label);
    
    input_img1 = single(input_img1(:, :, :)) - IMAGE_MEAN;
    input_img2 = single(input_img2(:, :, :)) - IMAGE_MEAN;
    input_img3 = single(input_img3(:, :, :)) - IMAGE_MEAN;
    input_img4 = single(input_img4(:, :, :)) - IMAGE_MEAN;
    
    input_img1 = permute(single(input_img1), [2, 1, 3]);
    input_img2 = permute(single(input_img2), [2, 1, 3]);
    input_img3 = permute(single(input_img3), [2, 1, 3]);
    input_img4 = permute(single(input_img4), [2, 1, 3]);
    
    net_inputs = {input_img1, input_img2, input_img3, input_img4};
    
    net.set_phase('test');
    net.reshape_as_input(net_inputs);
    scores = net.forward(net_inputs);
    scores = scores{1};
    net_inputs = {input_img1(end:-1:1, :, :), input_img2(end:-1:1, :, :), input_img3(end:-1:1, :, :), input_img4(end:-1:1, :, :)};
    scores2 = net.forward(net_inputs);
    scores = scores + scores2{1}(end:-1:1, :, :);
    scores = permute(scores, [2, 1, 3]);
    scores = imresize(scores, [256, 256]);
    [~, result] = max(scores, [], 3);
    result = uint8(result - 1);
    
    if (exist('show', 'var'))
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
        subplot(326);
        imshow(result, cmap);
        drawnow;
    end;
    cmap = VOClabelcolormap(7);
    imwrite(result, ['/DATA3/caffe/data/carotidmri/result/', label_list{idx}]);
    result = cmap(result +1, :) * 255;
    result = reshape(result, [256, 256, 3]);
    label = cmap(label +1, :) * 255;
    label = reshape(label, [256, 256, 3]);
    view = cat(1, cat(2, img1, img2),  cat(2, img3, img4),  cat(2, label, result));
    imwrite(view, ['/DATA3/caffe/data/carotidmri/result/view_', label_list{idx}]);
end;
