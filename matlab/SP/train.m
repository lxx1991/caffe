clear; clc;

vlfeat_root = fullfile('/', 'DATA3', 'vlfeat-0.9.20', 'toolbox', 'vl_setup');
run(vlfeat_root);

if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 1;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);
%%

dir_model = fullfile('..', '..', 'examples', 'SP');
file_solver = fullfile(dir_model, 'SP_solver.prototxt');

caffe_solver = caffe.Solver(file_solver);

dir_dataset = fullfile('..', '..', 'data', 'VOC_arg_instance');
train_list = fullfile(dir_dataset, 'train.txt');
fid = fopen(train_list);
name_list=textscan(fid, '%s');
name_list=name_list{1};
fclose(fid);

IMAGE_DIM = 512;
d = load('vgg16_mean');
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1;
REGIONSIZE = 20; 
REGULARIZER = 0.1;

xy = cat(3, repmat(1:512, 512, 1), repmat([1:512]', 1, 512));
%%
while (caffe_solver.iter() < caffe_solver.max_iter())
    tic;
        if (~exist(fullfile(dir_dataset, 'img', [name_list{idx}, '.jpg']), 'file'))
            idx = mod(idx, length(name_list)) +1;
            continue;
        end;
        img = imread(fullfile(dir_dataset, 'img', [name_list{idx}, '.jpg']));
        [ sp, o ] = segment_slic(img, img, REGIONSIZE, REGULARIZER);
        sp_edge = gen_pairs(double(sp), sp);

        height_img = size(img, 1);
        width_img = size(img, 2);

        input_img = single(img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
        input_xy = single(xy(1:height_img, 1:width_img, :));
        sp_label = zeros(max(sp(:)) + 1, 1, 'single');
        sp_edge(:, 1) = sp_edge(:, 1) + (double(max(sp(:))) + 1);
        input_sp = single(sp);

        input_img = permute(cat(3, single(input_img), single(input_xy)), [2, 1, 3]);
        input_xy = permute(single(input_xy * REGULARIZER*3), [2, 1, 3]);
        input_sp = permute(single(input_sp), [2, 1, 3]);
        sp_label = permute(single(sp_label), [2, 1, 3]);
        sp_edge = permute(single(sp_edge), [2, 1, 3]);


        net_inputs = {input_img, input_xy, input_sp, sp_label, sp_edge};
        caffe_solver.net.set_phase('train');
        caffe_solver.net.reshape_as_input(net_inputs);
        caffe_solver.net.set_input_data(net_inputs);
        caffe_solver.step_sample();
    toc;
    if (mod(idx, 10) == 0)
        subplot(121);
        imshow(o);
        subplot(122);
        feat = caffe_solver.net.blobs('L2norm').get_data();
        feat = permute(feat, [2, 1, 3]);
        [ sp, o ] = segment_slic(img, feat, REGIONSIZE, REGULARIZER*3);
        imshow(o);
        drawnow;
    end;
    idx = mod(idx, length(name_list)) +1;
end;
