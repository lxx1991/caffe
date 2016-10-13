clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;

caffe.reset_all();

use_gpu = 1;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);

dir_model = fullfile('..', '..', 'examples', 'IRes_VOC_Mapping');
file_solver = fullfile(dir_model, 'IRes_VOC_LRN_Mat_solver.prototxt');
file_weight = fullfile(dir_model, 'model_LRN', 'unary_mat_iter_0.caffemodel');

caffe_solver = caffe.Solver(file_solver);
caffe_solver.net.copy_from(file_weight);

%%
addpath(fullfile('..', '..', 'data', 'VOCcode'));
VOCinit;
class_hash = containers.Map();
for i = 1:VOCopts.nclasses
    class_hash(VOCopts.classes{i}) = i;
end;

dir_dataset = fullfile('..', '..', 'data', 'VOC_arg');
train_list = fullfile(dir_dataset, 'train1.txt');

[name_list, label_list, useless]= textread(train_list, '%s %s %s');

if exist('./object.mat', 'file')
    load('./object.mat')
else
    object = cell(1, 20);

    for idx = 1:length(name_list)
        if (~exist(fullfile(dir_dataset, 'Annotations', [name_list{idx}(13:end-3) 'xml']), 'file'))
            continue;
        end;
        recs=PASreadrecord(fullfile(dir_dataset, 'Annotations', [name_list{idx}(13:end-3) 'xml']));
        img = imread(fullfile(dir_dataset, name_list{idx}));
        [label, cmap] = imread(fullfile(dir_dataset, label_list{idx}));
        for j = 1:length(recs.objects)
            bb = recs.objects(j).bbox;
            clsinds=class_hash(recs.objects(j).class);
            patch = label(bb(2):bb(4), bb(1):bb(3));
            s = sum(sum(patch == clsinds));
            if s*2 > length(patch(:))
                sample.bb = bb;
                sample.img = idx;
                object{clsinds} = [object{clsinds}; sample];
            end;
        end;
    end;
end;




%%

IMAGE_DIM = 512;

d = load('vgg16_mean');
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1; show = 10;

%%
while (caffe_solver.iter() <= caffe_solver.max_iter())
    
    cls = randi(length(object));
    
    p = zeros(2, 1);
    p(1) = randi(length(object{cls}));
    p(2) = randi(length(object{cls}));
    while  p(2) ==  p(1)
        p(2) = randi(length(object{cls}));
    end;
    
    net_inputs = cell(1, 6);
    img = cell(1,2);
    label = cell(1,2);
    roi = cell(1,2);
    
    for i = 1:2
        idx = object{cls}(p(i)).img;

        img{i} = imread(fullfile(dir_dataset, name_list{idx}));
        [label{i}, cmap] = imread(fullfile(dir_dataset, label_list{idx}));
        roi{i} = object{cls}(p(i)).bb;
        
        [img{i}, label{i}, roi{i}] = im_tf( img{i}, label{i}, roi{i});
        height_img = size(img{i}, 1);
        width_img = size(img{i}, 2);
        
        imshow(img{i}(roi{i}(2) + 1:roi{i}(4), roi{i}(1) + 1:roi{i}(3), :));

        net_inputs{(i-1)*3+1} = permute(single(img{i}(:, :, [3, 2, 1])) - IMAGE_MEAN(1:height_img, 1:width_img, :), [2, 1, 3]);
        net_inputs{(i-1)*3+2} = permute(single([0, roi{i}]), [2, 1, 3]);
        net_inputs{(i-1)*3+3} = permute(single(label{i}), [2, 1, 3]);
    end;
    
    
    caffe_solver.net.set_phase('train');
    caffe_solver.net.reshape_as_input(net_inputs);
    
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step_sample();
    
    if (exist('show', 'var') && mod(caffe_solver.iter(), show) == 0)
        
        colormap default;
        subplot(427);
        feat = permute(caffe_solver.net.blobs('mapping').get_data(), [2, 1, 3]);
        for i = 1:7
            for j = 1:7
                norm_feat(i, j) = norm(reshape(feat(i, j, :), 2048, 1));
            end;
        end
        imagesc(norm_feat / norm(norm_feat));
        axis image;
        freezeColors;
        
        colormap default;
        subplot(428);
        feat = permute(caffe_solver.net.blobs('roi_pool2').get_data(), [2, 1, 3]);
        for i = 1:7
            for j = 1:7
                norm_feat(i, j) = norm(reshape(feat(i, j, :), 2048, 1));
            end;
        end
        imagesc(norm_feat / norm(norm_feat));
        axis image;
        freezeColors;
        
        for i = 1:2
            subplot(4, 2, i);
            imshow(img{i});
            rectangle('Position', [roi{i}(1)+1,roi{i}(2)+1,roi{i}(3)-roi{i}(1),roi{i}(4) - roi{i}(2)], 'LineWidth', 2);
            subplot(4, 2, 2 + i);
            imshow(label{i}, cmap);
        end;
        subplot(425);
        score = permute(caffe_solver.net.blobs('upscore').get_data(), [2, 1, 3, 4]);
        [~, result] = max(score, [], 3);
        imshow(result(:, :, :), cmap);
        
        subplot(426);
        score = permute(caffe_solver.net.blobs('v2_upscore').get_data(), [2, 1, 3, 4]);
        [~, result] = max(score, [], 3);
        imshow(result(:, :, :), cmap);
       
        
        drawnow;
    end;
end;
