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
file_solver = fullfile(dir_model, 'solver.prototxt');
file_weight = fullfile(dir_model, 'VGG_ILSVRC_16_layers_conv.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);


dir_dataset = fullfile('..', '..', 'data', 'carotidmri');
train_list = fullfile(dir_dataset, 'test_list');
dir_dataset = fullfile(dir_dataset, 'images');

[name_list1, name_list2, name_list3, name_list4, label_list]= textread(train_list, '%s %s %s %s %s');

IMAGE_DIM = 512;
d = load('vgg16_mean');
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1;
%%
while (caffe_solver.iter() < caffe_solver.max_iter())
    
    if (~exist(fullfile(dir_dataset, [name_list1{idx}]), 'file'))
        idx = mod(idx,length(name_list1)) + 1;
        continue;
    end;
    
    img = repmat(imread(fullfile(dir_dataset, name_list1{idx})), 1, 1, 3);
    img = cat(3, img, repmat(imread(fullfile(dir_dataset, name_list2{idx})), 1, 1, 3));
    img = cat(3, img, repmat(imread(fullfile(dir_dataset, name_list3{idx})), 1, 1, 3));
    img = cat(3, img, repmat(imread(fullfile(dir_dataset, name_list4{idx})), 1, 1, 3));
    label = imread(fullfile(dir_dataset, label_list{idx}));

    new_height_img = 320;
    new_width_img = 320;
    
    
    input_img = imresize(img, [new_height_img, new_width_img]);
   
    input_label = imresize(label, [new_height_img, new_width_img], 'nearest');

    input_img = single(input_img(:, :, :)) - repmat(IMAGE_MEAN(1:new_height_img, 1:new_width_img, :), 1, 1, 4);
    
    input_img = permute(single(input_img), [2, 1, 3]);
    input_label = permute(single(input_label), [2, 1, 3]);
    net_inputs = {input_img(:, :, 1:3), input_img(:, :, 4:6), input_img(:, :, 7:9), input_img(:, :, 10:12), input_label};
    
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    %tic;
    caffe_solver.step_sample();
    %toc;
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
    idx = mod(idx,length(name_list1)) + 1;
end;
