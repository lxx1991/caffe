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
file_model = fullfile(dir_model, 'SP_test.prototxt');
file_weight = fullfile(dir_model, 'model', 'unary_iter_8000.caffemodel');


net = caffe.Net(file_model, file_weight, 'test');

dir_dataset = fullfile('..', '..', 'data', 'VOC_arg_instance');
train_list = fullfile(dir_dataset, 'val.txt');
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
while (idx < length(name_list))
    tic;
        if (~exist(fullfile(dir_dataset, 'img', [name_list{idx}, '.jpg']), 'file'))
            idx = idx + 1;
            continue;
        end;
        img = imread(fullfile(dir_dataset, 'img', [name_list{idx}, '.jpg']));
        [ sp, o ] = segment_slic(img, img, REGIONSIZE, REGULARIZER);
        %sp_edge = gen_pairs(double(sp), sp);

        height_img = size(img, 1);
        width_img = size(img, 2);

        input_img = single(img(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :);
        input_xy = single(xy(1:height_img, 1:width_img, :));
        sp_label = zeros(max(sp(:)) + 1, 1, 'single');
        %sp_edge(:, 1) = sp_edge(:, 1) + (double(max(sp(:))) + 1);
        input_sp = single(sp);

        input_img = permute(cat(3, single(input_img), single(input_xy)), [2, 1, 3]);
        input_xy = permute(single(input_xy * REGULARIZER), [2, 1, 3]);
        input_sp = permute(single(input_sp), [2, 1, 3]);
        sp_label = permute(single(sp_label), [2, 1, 3]);
        %sp_edge = permute(single(sp_edge), [2, 1, 3]);


        net_inputs = {input_img};
        net.reshape_as_input(net_inputs);
        output = net.forward(net_inputs);
    
    
        subplot(121);
        imshow(o);
        imwrite(o, ['result/', num2str(idx), '_slic.jpg']);
        subplot(122);
        feat = net.blobs('L2norm').get_data();
        feat = permute(feat, [2, 1, 3]);
        [ sp, o ] = segment_slic(img, feat, REGIONSIZE, REGULARIZER);
        imshow(o);
        drawnow;
        imwrite(o, ['result/', num2str(idx), '_ours.jpg']);
        idx = idx +1;
end;
