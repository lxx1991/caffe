clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 1;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);


dir_dataset = fullfile('..', '..', 'data', 'VOC_arg_instance');
train_list = fullfile(dir_dataset, 'train.txt');
fid = fopen(train_list);
name_list=textscan(fid, '%s');
name_list=name_list{1};
fclose(fid);

IMAGE_DIM = 512;
d = load('vgg16_mean');
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

%% mask score

dir_model = fullfile('..', '..', 'examples', 'RIS', 'init_model');
file_net = fullfile(dir_model, 'DPN.prototxt');
file_weight = fullfile(dir_model, 'init.caffemodel');
net = caffe.Net(file_net, file_weight, 'test');

mask_idx = zeros(length(name_list), 1);
show = false;

for idx = 1:length(name_list)
    disp(idx);
    if (~exist(fullfile(dir_dataset, 'inst', [name_list{idx}, '.mat']), 'file'))
        continue;
    end;
    load(fullfile(dir_dataset, 'inst', name_list{idx}));
    tmp = find(GTinst.Categories == 15);
    if isempty(tmp)
        mask_idx(idx) = -1;
    else
        img = imread(fullfile(dir_dataset, 'img', [name_list{idx}, '.jpg']));

        height_img = size(img, 1);
        width_img = size(img, 2);
        st_h = floor((IMAGE_DIM - height_img) / 2);
        st_w = floor((IMAGE_DIM - width_img) / 2);

        input_img = zeros(IMAGE_DIM, IMAGE_DIM, 3, 'single');
        input_img(st_h+1:st_h+height_img, st_w+1:st_w+width_img, :) = single(img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
        net_inputs = {permute(input_img, [2, 1, 3])};
        net.reshape_as_input(net_inputs);
        output = net.forward(net_inputs);
        
        output = permute(output{1}, [2, 1, 3]);
        output = output(st_h+1:st_h+height_img, st_w+1:st_w+width_img, 16);
        
        cnt = -Inf;
        for i = 1:length(tmp)
            y = mean(output(GTinst.Segmentation == tmp(i)));
            if (y > cnt)
                cnt = y;
                mask_idx(idx) = tmp(i);
            end;
        end;
        if (show)
            subplot(221);
            imshow(img);
            subplot(222);
            imagesc(GTinst.Segmentation);
            axis image;
            subplot(223);
            imagesc(output);
            axis image;
            subplot(224);
            temp = GTinst.Segmentation;
            temp(temp ~= mask_idx(idx)) = 0;
            imagesc(temp);
            axis image;
            drawnow;
        end;
    end;
end;

save('mask_idx', 'mask_idx', '-v7.3');

%% training
dir_model = fullfile('..', '..', 'examples', 'RIS');
file_solver = fullfile(dir_model, 'RIS_weak_solver.prototxt');
file_weight = fullfile(dir_model, 'init_model', 'init.caffemodel');

caffe.reset_all();
caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);

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
    
    if (mask_idx(idx) == -1)
        input_mask = zeros(IMAGE_DIM, IMAGE_DIM, 1, 'single');
    else
        input_mask = zeros(IMAGE_DIM, IMAGE_DIM, 1, 'single');
        input_mask(st_h+1:st_h+height_img, st_w+1:st_w+width_img, 1) = single(GTinst.Segmentation == mask_idx(idx));
    end;
    
    input_img = permute(input_img, [2, 1, 3]);
    input_mask = permute(input_mask, [2, 1, 3]);
    net_inputs = {input_img, input_mask};
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    output = caffe_solver.net.get_output();
    %if (show > 0 && mod(caffe_solver.iter(), show) == 0)
    if (length(find(GTinst.Categories == 15)) >= 2)
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
