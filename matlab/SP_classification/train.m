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
dir_model = fullfile('..', '..', 'examples', 'SP_classification');
file_solver = fullfile(dir_model, 'SP_classification_solver.prototxt');
file_weight = fullfile(dir_model, 'VGG_ILSVRC_16_layers_conv.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);


dir_dataset = fullfile('..', '..', 'data', 'VOC_arg_instance');
train_list = fullfile(dir_dataset, 'train.txt');
fid = fopen(train_list);
name_list=textscan(fid, '%s');
name_list=name_list{1};
fclose(fid);

IMAGE_DIM = 512;
d = load('vgg16_mean');
cmap = VOClabelcolormap();
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1; show = 10;
%%
while (caffe_solver.iter() < caffe_solver.max_iter())
    
    if (~exist(fullfile(dir_dataset, 'cls', [name_list{idx}, '.mat']), 'file'))
        idx = mod(idx,length(name_list)) + 1;
        continue;
    end;
    
    load(fullfile(dir_dataset, 'cls', name_list{idx}));
    img = imread(fullfile(dir_dataset, 'img', [name_list{idx}, '.jpg']));
    sp = imread(fullfile(dir_dataset, 'superpixel_20_0.1', [name_list{idx}, '.png']));

    
    height_img = size(img, 1);
    width_img = size(img, 2);
    new_height_img = ceil(height_img / 32) * 32;
    new_width_img = ceil(width_img / 32) * 32;
    
    
    input_img = imresize(img, [new_height_img, new_width_img]);
    input_sp = imresize(sp, [new_height_img, new_width_img], 'nearest');
    input_label = imresize(GTcls.Segmentation, [new_height_img, new_width_img], 'nearest');

    input_img = single(input_img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:new_height_img, 1:new_width_img, :);
    
    sp_num = max(sp(:)) + 1;
    
    sp_label = zeros(sp_num, 1, 1);
    
    
    for i = 0 : sp_num - 1
        sp_labels = input_label(input_sp == i);
        [sp_label(i+1), feq]= mode(sp_labels);
        %ignore label
        if feq <= length(sp_labels) / 2
            sp_label(i+1) = 255;
        end;
    end;
    

    input_img = permute(single(input_img), [2, 1, 3]);
    input_sp = permute(single(input_sp), [2, 1, 3]);
    sp_label = permute(single(sp_label), [2, 1, 3]);
    net_inputs = {input_img, input_sp, sp_label};
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    tic;
    caffe_solver.step(1);
    toc;
    if (exist('show', 'var') && mod(caffe_solver.iter(), show) == 0)
        subplot(221);
        imshow(img);
        subplot(222);
        imshow(GTcls.Segmentation, cmap);
        score = caffe_solver.net.blobs('score').get_data();
        [~, score] = max(permute(score, [2, 1, 3]), [], 3);
        subplot(223);
        result = zeros(size(sp), 'uint8');
        for i = 0 : sp_num - 1
            result(sp == i) = score(i + 1) - 1;
        end;
        imshow(result, cmap);
        drawnow;
    end;
    idx = mod(idx,length(name_list)) + 1;
end;
