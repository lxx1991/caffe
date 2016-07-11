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
file_model = fullfile(dir_model, 'SP_test.prototxt');
file_weight = fullfile(dir_model, 'model', 'unary_iter_6000.caffemodel');

net = caffe.Net(file_model, file_weight, 'test');

dir_dataset = fullfile('..', '..', 'data', 'VOC_arg_instance');
train_list = fullfile(dir_dataset, 'val.txt');
fid = fopen(train_list);
name_list=textscan(fid, '%s');
name_list=name_list{1};
fclose(fid);

IMAGE_DIM = 512;
d = load('vgg16_mean');
cmap = VOClabelcolormap();
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1; show = 1;
%%
while (idx <= length(name_list))
    
    if (~exist(fullfile(dir_dataset, 'cls', [name_list{idx}, '.mat']), 'file'))
        idx = idx + 1;
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
    net.reshape_as_input(net_inputs);
    tic;
    output = net.forward(net_inputs);
    toc;

    subplot(221);
    imshow(img);
    subplot(222);
    imshow(GTcls.Segmentation, cmap);
    score = net.blobs('score').get_data();
    [~, score] = max(permute(score, [2, 1, 3]), [], 3);
    subplot(223);
    result = zeros(size(sp), 'uint8');
    for i = 0 : sp_num - 1
        result(sp == i) = score(i + 1) - 1;
    end;
    imshow(result, cmap);
    drawnow;
    pause;

    idx = idx + 1;
end;
