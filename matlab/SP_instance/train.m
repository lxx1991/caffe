clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 1;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);

%%
dir_model = fullfile('..', '..', 'examples', 'SP_instance');
file_solver = fullfile(dir_model, 'SP_instance_solver.prototxt');
file_weight = fullfile(dir_model, 'unary_iter_9000.caffemodel');

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

idx = 1; show = 10; step = 3;
%%
while (caffe_solver.iter() < caffe_solver.max_iter())
    
    tic;
    
    if (~exist(fullfile(dir_dataset, 'cls', [name_list{idx}, '.mat']), 'file'))
        idx = mod(idx,length(name_list)) + 1;
        continue;
    end;
    
    load(fullfile(dir_dataset, 'inst', name_list{idx}));
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
    input_instance = imresize(GTinst.Segmentation, [new_height_img, new_width_img], 'nearest');
   
    input_img = single(input_img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:new_height_img, 1:new_width_img, :);
    
    [sp_edge, sp_label, sp_instance] = sp_graph(double(input_sp), step, double(input_label), double(input_instance));
    
    if isempty(find(sp_edge(:, 3) == 0, 1))
        idx = mod(idx,length(name_list)) + 1;
        continue;
    end;
    
    input_img = permute(single(input_img), [2, 1, 3]);
    input_sp = permute(single(input_sp), [2, 1, 3]);
    label = permute(single(input_label), [2, 1, 3]);
    sp_label = permute(single(sp_label), [2, 1, 3]);
    edge_label = permute(single(sp_edge(:, 3)), [3, 2, 4 ,1]);
    sp_edge = permute(single(sp_edge(:, 1:2)), [2, 1, 3]);
    
    
    net_inputs = {input_img, input_sp, label, sp_label, sp_edge, edge_label};
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step_sample();
    toc;
    if (exist('show', 'var') && mod(caffe_solver.iter(), show) == 0)
        subplot(221);
        imshow(img);
        subplot(222);
        imshow(GTcls.Segmentation, cmap);
        score = caffe_solver.net.blobs('upscore').get_data();
        [~, score] = max(permute(score, [2, 1, 3]), [], 3);
        subplot(223);
        imshow(score, cmap);
        subplot(224);
        if (size(sp_edge, 2)~=1)
            fc10 = caffe_solver.net.blobs('fc10').get_data();
            d0 = fc10(1, edge_label(:) == 0) - fc10(2, edge_label(:) == 0);
            d1 = fc10(1, edge_label(:) == 1) - fc10(2, edge_label(:) == 1);
            subplot(426);
            hist(d0(:), -5:0.2:2);
            subplot(428);
            hist(d1(:), -5:0.2:2);
        end;
        drawnow;
    end;
    idx = mod(idx,length(name_list)) + 1;
end;
