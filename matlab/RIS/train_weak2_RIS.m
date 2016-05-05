clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 0;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);

%%
dir_model = fullfile('..', '..', 'examples', 'RIS');
file_solver = fullfile(dir_model, 'RIS_weak2_solver.prototxt');
file_weight = fullfile(dir_model, 'init_model', 'init.caffemodel');

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
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

%%
idx = 1;
show = 20;
%%
while (caffe_solver.iter() < caffe_solver.max_iter())
    if (~exist(fullfile(dir_dataset, 'inst', [name_list{idx}, '.mat']), 'file'))
        idx = mod(idx,length(name_list)) + 1;
        continue;
    end;
    load(fullfile(dir_dataset, 'inst', name_list{idx}));
    img = imread(fullfile(dir_dataset, 'img', [name_list{idx}, '.jpg']));
    
    height_img = size(img, 1);
    width_img = size(img, 2);
    st_h = floor((IMAGE_DIM - height_img) / 2);
    st_w = floor((IMAGE_DIM - width_img) / 2);
    
    input_img = zeros(IMAGE_DIM, IMAGE_DIM, 3, 'single');
    input_img(st_h+1:st_h+height_img, st_w+1:st_w+width_img, :) = single(img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
    
    tmp = find(GTinst.Categories == 15);
    if length(tmp) < 1
        input_mask = zeros(IMAGE_DIM, IMAGE_DIM, 1, 'single');
    else
        input_mask = zeros(IMAGE_DIM, IMAGE_DIM, 1, 'single');
        cnt = 0;
        for i = 1:length(tmp)
            y = sum(sum(GTinst.Segmentation == tmp(i)));
            if y > cnt
                cnt = y;
                input_mask(st_h+1:st_h+height_img, st_w+1:st_w+width_img, 1) = single(GTinst.Segmentation == tmp(i));
            end;
        end;
    end;
    % permute x,y axis
    input_img = permute(input_img, [2, 1, 3]);
    input_mask = permute(input_mask, [2, 1, 3]);
    net_inputs = {input_img, input_mask};
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    output = caffe_solver.net.get_output();
    %if (show > 0 && mod(caffe_solver.iter(), show) == 0)
    if (length(tmp) >= 2)
        subplot(221);
        imshow(img);
        subplot(222);
        imagesc(permute(input_mask(st_w+1:st_w+width_img, st_h+1:st_h+height_img, 1), [2, 1, 3]), [0, 1]);
        axis image;
        title(num2str(length(tmp)));
        pro_map = caffe_solver.net.blobs('upscore').get_data();
        pro_map = permute(pro_map, [2, 1, 3]);
        subplot(223);
        pro_map = exp(pro_map)./ repmat(sum(exp(pro_map), 3), [1, 1, 2]);
        imagesc(pro_map(st_h+1:st_h+height_img, st_w+1:st_w+width_img, 2), [0, 1]);
        axis image;
        drawnow;
    end;
    idx = mod(idx,length(name_list)) + 1;
end;
