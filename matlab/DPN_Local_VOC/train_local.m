clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 5;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);
%  caffe.set_mode_cpu();

%%
dir_model = fullfile('..', '..', 'examples', 'DPN_Local_VOC');
file_solver = fullfile(dir_model, 'DPN_Local_VOC_solver.prototxt');
file_weight = fullfile(dir_model, 'VGG_ILSVRC_16_layers_conv.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);


dir_dataset = fullfile('..', '..', 'data', 'VOC_arg');
train_list = fullfile(dir_dataset, 'train.txt');

[name_list, label_list, edge_list]= textread(train_list, '%s %s %s');

IMAGE_DIM = 512;
d = load('vgg16_mean');
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1;
%%
while (caffe_solver.iter() <= caffe_solver.max_iter())
    
    if (~exist(fullfile(dir_dataset, [edge_list{idx}]), 'file'))
        idx = mod(idx,length(name_list)) + 1;
        continue;
    end;
    
    load(fullfile(dir_dataset, edge_list{idx}));
    img = imread(fullfile(dir_dataset, name_list{idx}));
    [label, cmap] = imread(fullfile(dir_dataset, label_list{idx}));

    
    height_img = size(img, 1);
    width_img = size(img, 2);
    new_height_img = floor(height_img / 32) * 32;
    new_width_img = floor(width_img / 32) * 32;
    
    
    input_img = img(1:new_height_img, 1:new_width_img, :);
    input_edge_map = edge_map(1:(new_height_img/8), 1:(new_width_img/8), :);
    input_label = label(1:new_height_img, 1:new_width_img, :);

    input_img = single(input_img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:new_height_img, 1:new_width_img, :);
    
    input_img = permute(single(input_img), [2, 1, 3]);
    input_edge_map = permute(single(input_edge_map), [2, 1, 3]);
    input_label = permute(single(input_label), [2, 1, 3]);
    net_inputs = {input_img, input_edge_map, input_label};
    
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step_sample();
    if (exist('show', 'var') && mod(caffe_solver.iter(), show) == 0)
        subplot(221);
        imshow(img);
        subplot(222);
        imshow(label, cmap);
        score = caffe_solver.net.blobs('upscore').get_data();
        [~, result] = max(permute(score, [2, 1, 3]), [], 3);
        subplot(223);
        imshow(result, cmap);
        drawnow;
    end;
    idx = mod(idx,length(name_list)) + 1;
end;
