clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 0;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);

%%
dir_model = fullfile('..', '..', 'examples', 'IRes_Cascade');
file_solver = fullfile(dir_model, 'IRes_Cascade_Mat_solver.prototxt');
file_weight = fullfile(dir_model, 'unary_mat_iter_18000.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);


dir_dataset = fullfile('..', '..', 'data', 'VOC_arg');
train_list = fullfile(dir_dataset, 'train1.txt');

[name_list, label_list, useless]= textread(train_list, '%s %s %s');

IMAGE_DIM = 640;

d = load('vgg16_mean');
IMAGE_MEAN = single(imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest'));

idx = 1; show = 100;
%%
while (caffe_solver.iter() <= caffe_solver.max_iter())
    
    if (idx == 1)
        idxs = randperm(length(name_list));
        name_list = name_list(idxs);
        label_list = label_list(idxs);
    end;
    
    img = imread(fullfile(dir_dataset, name_list{idx}));
    [label, cmap] = imread(fullfile(dir_dataset, label_list{idx}));
    [img, label] = im_tf(img, label);
    height_img = size(img, 1);
    width_img = size(img, 2);

    input_img = permute(single(img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :), [2, 1, 3]);
    input_label = permute(single(label), [2, 1, 3]);
    net_inputs = {input_img, input_label};
    
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step_sample();
    
    if (exist('show', 'var') && mod(caffe_solver.iter(), show) == 0)
        subplot(321);
        imshow(img);
        subplot(322);
        imshow(label, cmap);
        subplot(323);
        score = caffe_solver.net.blobs('cascade1_upscore').get_data();
        [~, result] = max(permute(score, [2, 1, 3]), [], 3);
        imshow(result, cmap);
        subplot(324);
        score = caffe_solver.net.blobs('cascade2_upscore').get_data();
        [~, result] = max(permute(score, [2, 1, 3]), [], 3);
        imshow(result, cmap);
        subplot(325);
        score = caffe_solver.net.blobs('cascade3_upscore').get_data();
        [~, result] = max(permute(score, [2, 1, 3]), [], 3);
        imshow(result, cmap);
        subplot(326);
        score = caffe_solver.net.blobs('upscore').get_data();
        [~, result] = max(permute(score, [2, 1, 3]), [], 3);
        imshow(result, cmap);
        drawnow;
    end;
    idx = mod(idx,length(name_list)) + 1;
end;
