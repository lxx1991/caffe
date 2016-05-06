% This script contains pipeline of FCN + DenseCRF
clear; clc;
run('/DATA3/vlfeat-0.9.20/toolbox/vl_setup');
if exist('../../+caffe', 'dir')
  addpath('../../');
end;
caffe.reset_all();
addpath(genpath('/DATA3/caffe_DPN/data/VOCcode'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set Directory

data_set = 'VOC_arg';
list_name = 'val';

dir_data = ['/DATA3/caffe_DPN/data/', data_set, '/'];
dir_img = [dir_data, 'JPEGImages/'];
file_list_name = [dir_data, 'ImageSets/Segmentation/', list_name, '.txt'];
file_img_mean = './vgg16_mean.mat';

dir_model = '/DATA3/caffe/matlab/SP_classification/superpixel_pooling_test/';
file_model = [dir_model, 'init.caffemodel'];
file_def_model = [dir_model, 'test.prototxt'];

results_name = 'test';
dir_results = ['/DATA3/caffe_DPN/data/results/', data_set, '/Segmentation/', results_name, '_', list_name, '_cls/']; mkdir(dir_results);

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
    im = im(1:256, 1:256, :);
    I =  vl_rgb2xyz(im);  
    I_single = single(I);  
    sp = vl_slic(I_single, 20, 0.1);
    
    height_img = size(im, 1);
    width_img = size(im, 2);

    im = single(im(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);

    image = single(im);
    
    % permute x,y axis
    image = permute(image, [2, 1, 3]);
    input_data = {image, single(permute(sp, [2, 1, 3])), zeros(1, max(sp(:)) + 1, 1, 1, 'single')};

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

    [~, maxlabel] = max(scores, [], 3);
    maxlabel = uint8(maxlabel - 1);
    sp = maxlabel(sp+1);
    imagesc(sp);
    pause;
    disp(['Processing Img ', num2str(id_img), '...']);
end