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
dir_model = fullfile('..', '..', 'examples', 'IRes_VOC');
file_solver = fullfile(dir_model, 'IRes_VOC_Mat_ms_solver.prototxt');
file_weight = fullfile(dir_model, 'model/unary_mat_iter_18000.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);


dir_dataset = fullfile('..', '..', 'data', 'VOC_arg');
train_list = fullfile(dir_dataset, 'train1.txt');

[name_list, label_list, useless]= textread(train_list, '%s %s %s');

IMAGE_DIM = 640;

d = load('vgg16_mean');
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1; show = 100; prev = 0; loss = 0;
losses = []; cnt = 0;
%%
while (caffe_solver.iter() <= caffe_solver.max_iter())
    
%     if (~exist(fullfile(dir_dataset, [edge_list{idx}]), 'file'))
%         idx = mod(idx,length(name_list)) + 1;
%         continue;
%     end;
    
    img = imread(fullfile(dir_dataset, name_list{idx}));
    [label, cmap] = imread(fullfile(dir_dataset, label_list{idx}));
    tic;
    [input_img, input_label] = im_tf(img, label);
    toc;
    height_img = size(input_img, 1);
    width_img = size(input_img, 2);

    input_img = single(input_img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
    
    input_img = permute(single(input_img), [2, 1, 3]);
    input_label = permute(single(input_label), [2, 1, 3]);
    net_inputs = {input_img, input_label};
    
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
        subplot(224);
        if caffe_solver.iter() ~= prev
            prev = caffe_solver.iter();
            hold on;
            plot(prev, min(1.5, loss/(show*10)), 'r.');
            cnt = cnt + 1;
            losses(cnt) = loss/(show*10);
            loss = 0;
        end;
        drawnow;
    end;
    loss = loss + caffe_solver.net.blobs('loss').get_data();
    idx = mod(idx,length(name_list)) + 1;
end;
