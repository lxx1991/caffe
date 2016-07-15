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

dir_model = '../../examples/Res_Coco/';
file_model = [dir_model, 'model/finetune2_iter_16000.caffemodel'];
file_def_model = [dir_ model, 'test/DPN2_VOC_test.prototxt'];

results_name = 'Res_VOC';
dir_results = ['../../data/results/', data_set, '/Segmentation/', results_name, '_', list_name, '_cls/']; mkdir(dir_results);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization;

use_gpu = 0;

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

    height_img = size(im, 1);
    width_img = size(im, 2);
    st_h = floor((IMAGE_DIM - height_img) / 2);
    st_w = floor((IMAGE_DIM - width_img) / 2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % prepare input (FCN)
    % input_data is Height x Width x Channel x Num
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = single(im(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
    % pad to fixed input size
    image = zeros(IMAGE_DIM, IMAGE_DIM, 3, 'single');
    image(st_h+1:st_h+height_img, st_w+1:st_w+width_img, :) = im;
    
    % permute x,y axis
    image = permute(image, [2, 1, 3]);
    input_data = {image};

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    tic;

    % init caffe network (spews logging info)
    
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
    scores = scores(st_h+1:st_h+height_img, st_w+1:st_w+width_img, :);

    [~, maxlabel] = max(scores, [], 3);
    maxlabel = uint8(maxlabel - 1);
    
    name_label_cur = [img_list{id_img}, '.png'];
    imwrite(uint8(maxlabel), cmap, [dir_results, name_label_cur]);

    disp(['Processing Img ', num2str(id_img), '...']);
end
VOCinit();

VOCevalseg(VOCopts, results_name);
caffe.reset_all();