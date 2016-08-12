clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;

rand('state', 0);

vlfeat_root = fullfile('/', 'DATA3', 'vlfeat-0.9.20', 'toolbox', 'vl_setup');
run(vlfeat_root);

REGIONSIZE = 20; 
REGULARIZER = 0.1;

caffe.reset_all();

use_gpu = 1;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);
%  caffe.set_mode_cpu();

%%
dir_model = fullfile('..', '..', 'examples', 'Res_SP_VOC');
file_solver = fullfile(dir_model, 'Res_SP_Local_Con_VOC_solver.prototxt');
file_weight = fullfile(dir_model, 'unary_mat_iter_9000.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);


dir_dataset = fullfile('..', '..', 'data', 'VOC_arg');
train_list = fullfile(dir_dataset, 'train1.txt');

[name_list, label_list, useless]= textread(train_list, '%s %s %s');

IMAGE_DIM = 640;

d = load('vgg16_mean');
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1; show = 100;
%%
while (caffe_solver.iter() <= caffe_solver.max_iter())

    disp(idx);

%     if (~exist(fullfile(dir_dataset, [edge_list{idx}]), 'file'))
%         idx = mod(idx,length(name_list)) + 1;
%         continue;
%     end;
    
    img = imread(fullfile(dir_dataset, name_list{idx}));
    [label, cmap] = imread(fullfile(dir_dataset, label_list{idx}));
    [input_img, input_label] = im_tf(img, label);
    s_img = input_img; s_label = input_label;
    
    height_img = size(input_img, 1);
    width_img = size(input_img, 2);
    
    tic;
    fid = fopen('/DATA3/gSLICr/con.txt', 'wb');
    fwrite(fid, height_img, 'int');
    fwrite(fid, width_img, 'int');
    fwrite(fid, reshape(permute(input_img, [2, 1, 3]), height_img * width_img, [])', 'uint8');
    system('/DATA3/gSLICr/build/demo /DATA3/gSLICr/con.txt /DATA3/gSLICr/seg_con.pgm');
    input_sp = imread('/DATA3/gSLICr/seg_con.pgm');
    [sp_label, input_sp]= sp_graph(double(input_sp), double(input_label));
    filler = gen_local_filler(imresize(input_sp, 1/8, 'nearest'), 5);
    s_sp = input_sp;
    fclose(fid);
    toc;

%     input_sp = segment_slic(input_img, REGIONSIZE, REGULARIZER);
%     [sp_label, input_sp]= sp_graph(double(input_sp), double(input_label));
%     s_sp = input_sp;
    

    input_img = single(input_img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
    
    input_img = permute(single(input_img), [2, 1, 3]);
    input_label = permute(single(input_label), [2, 1, 3]);
    input_sp = permute(single(input_sp), [2, 1, 3]);
    filler = permute(single(filler), [2, 1, 3]);
    net_inputs = {input_img, input_label, filler};
    
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step_sample();
    
    if (exist('show', 'var') && mod(caffe_solver.iter(), show) == 0)
        subplot(221);
        imshow(s_img);
        subplot(222);
        imshow(s_label, cmap);
        score = caffe_solver.net.blobs('upscore').get_data();
        [~, result] = max(permute(score, [2, 1, 3]), [], 3);
        subplot(223);
        imshow(result, cmap);
        drawnow;
    end;
    idx = mod(idx,length(name_list)) + 1;
end;
