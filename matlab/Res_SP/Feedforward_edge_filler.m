
% This script contains pipeline of FCN + DenseCRF
clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();
addpath(genpath('../../data/VOCcode'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set Directory

data_set = 'VOC_arg';
list_name = 'val';

dir_data = ['../../data/', data_set, '/'];
dir_img = [dir_data, 'JPEGImages/'];
file_list_name = [dir_data, 'ImageSets/Segmentation/', list_name, '.txt'];
file_img_mean = './vgg16_mean.mat';

dir_model = '../../examples/Res_SP_VOC/';
file_model = [dir_model, 'model/unary_local_edge_iter_5000.caffemodel'];
file_def_model = [dir_model, 'test/Res_SP_Local_VOC.prototxt'];

results_name = 'Res_SP_Local_edge_VOC';
dir_results = ['../../data/results/', data_set, '/Segmentation/', results_name, '_', list_name, '_cls/']; mkdir(dir_results);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization;

use_gpu = 1;

if use_gpu ~= -1
    caffe.set_mode_gpu();
    caffe.set_device(use_gpu);
else
    caffe.set_mode_cpu();
end
net = caffe.Net(file_def_model, file_model, 'test');

d = load(file_img_mean);
IMAGE_DIM = 512;
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

img_list = textread(file_list_name, '%s');
num_img = length(img_list);

cmap = VOClabelcolormap(255);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FCN-8s + DenseCRF
for id_img = 1:num_img

    % read image
    name_img_cur = [img_list{id_img}, '.jpg'];
    im = imread([dir_img, name_img_cur]);
    input_label = imread(['/DATA3/caffe/data/VOC_arg/SegmentationClass/' img_list{id_img}, '.png']);
    
    pre_height_img = size(im, 1);
    pre_width_img = size(im, 2);
    height_img = round(pre_height_img/32) * 32;
    width_img = round(pre_width_img/32) * 32;
    im = imresize(im, [height_img, width_img]);
    input_label = imresize(input_label, [height_img, width_img], 'nearest');
    
    
%     tic;
%     filler = gen_edge_local_filler(imresize(input_label, 1/8, 'nearest'), 5);
%     toc
    
    tic;
    filler1 = gen_edge_local_filler(imresize(input_label, 1/2, 'nearest'), 5);
    filler2 = gen_edge_local_filler(imresize(input_label, 1/4, 'nearest'), 5);
    toc
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % prepare input (FCN)
    % input_data is Height x Width x Channel x Num
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = single(im(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
    
    % permute x,y axis
    input_img = permute(single(im), [2, 1, 3]);
    filler1 = permute(single(filler1), [2, 1, 3]);
    filler2 = permute(single(filler2), [2, 1, 3]);
    input_data = {input_img, filler1, filler2};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    tic;

    % init caffe network (spews logging info)
    net.reshape_as_input(input_data);
    scores = net.forward(input_data);
    scores = scores{1};
    
%     input_data = {image(end:-1:1, :, :)};
%     scores2 = caffe('forward', input_data);
%     scores = scores + scores2{1}(end:-1:1, :, :);
    
    toc;

    disp('VGG16 Done.');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % save results
    
    % permute and crop scores
    scores = permute(scores, [2, 1, 3]);
    scores = imresize(scores, [pre_height_img, pre_width_img]);
    
    [~, maxlabel] = max(scores, [], 3);
    maxlabel = uint8(maxlabel - 1);
    
    name_label_cur = [img_list{id_img}, '.png'];
    imwrite(uint8(maxlabel), cmap, [dir_results, name_label_cur]);

    disp(['Processing Img ', num2str(id_img), '...']);
end
VOCinit();

VOCevalseg(VOCopts, results_name);
caffe.reset_all();