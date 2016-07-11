clear; clc;
if exist('../+caffe', 'dir')
  addpath('..');
end;
caffe.reset_all();

use_gpu = 0;
caffe.set_mode_gpu();
caffe.set_device(use_gpu);
%%
dir_model = fullfile('..', '..', 'examples', 'SP_instance');
file_model = fullfile(dir_model, 'SP_instance_tri_test2.prototxt');
file_weight = fullfile(dir_model, 'model', 'unary_tri_iter_8000.caffemodel');

net = caffe.Net(file_model, file_weight, 'test');

dir_dataset = fullfile('..', '..', 'data', 'VOC_arg_instance');
train_list = fullfile(dir_dataset, 'val.txt');
fid = fopen(train_list);
name_list=textscan(fid, '%s');
name_list=name_list{1};
fclose(fid);

IMAGE_DIM = 512;
d = load('vgg16_mean');
cmap = VOClabelcolormap();
IMAGE_MEAN = imresize(d.image_mean, [IMAGE_DIM, IMAGE_DIM], 'nearest');

idx = 1; show = 1; step = 3;
%%
while (idx <= length(name_list))
    
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
    
    [sp_edge, sp_label, sp_instance] = sp_graph_tri(double(input_sp), step, double(input_label), double(input_instance));
    
    if isempty(sp_edge)
        idx = mod(idx,length(name_list)) + 1;
        continue;
    end;

    input_img = permute(single(input_img), [2, 1, 3]);
    input_sp = permute(single(input_sp), [2, 1, 3]);
    label = permute(single(input_label), [2, 1, 3]);
    sp_label = permute(single(sp_label), [2, 1, 3]);
    sp_edge = permute(single(sp_edge), [2, 1, 3]);
    
    
    net_inputs = {input_img, input_sp, label, sp_label, sp_edge};
    net.reshape_as_input(net_inputs);
    tic;
    output = net.forward(net_inputs);
    toc;

    subplot(221);
    imshow(img);
    subplot(222);
    imshow(GTinst.Segmentation + 1, cmap);
    score = net.blobs('upscore').get_data();
    [~, score] = max(permute(score, [2, 1, 3]), [], 3);
    subplot(223);
    imshow(score, cmap);

    subplot(224);
    if (size(sp_edge, 2)~=1)
%         feat_a = net.blobs('feat_a').get_data();
%         feat_n = net.blobs('feat_n').get_data();
%         feat_p = net.blobs('feat_p').get_data();
        feat = net.blobs('L2norm').get_data();
        feat = permute(feat, [2, 3, 1]);
        
        temp = accumarray(input_sp(:) +1, 1);
        g = [];
        for j = 1:size(sp_edge, 2)
            g = [g; sp_edge(:, j)];
        end;
        g = unique(g);
        
        c = zeros(length(g), size(feat, 1));
        feat_c = zeros(length(g), 256);
        for j = 1:length(g)
            c(j, g(j)) = 1;
            feat_c(j, :) = feat(g(j), :);
        end;
%        outputVideo = VideoWriter([num2str(idx), '.avi']);
%        outputVideo.FrameRate = 3;
%        open(outputVideo);
        while (size(c, 1) > 3)
            dist = pdist2(feat_c, feat_c) + diag(inf*ones(size(feat_c, 1), 1));
            [mindist, jj] = min(dist(:));
            kk = floor((jj-1) / size(feat_c, 1)) + 1;
            jj = mod(jj-1,  size(feat_c, 1)) + 1;
            
            disp(mindist);
            feat_c(jj, :) =  feat_c(jj, :) * (c(jj, :) * temp) + feat_c(kk, :) * (c(kk, :) * temp);
            c(jj, :)  = c(jj, :) + c(kk, :);
            feat_c(jj, :) =  feat_c(jj, :) / norm(feat_c(jj, :));
            
            c(kk, :) = c(end, :);
            feat_c(kk, :) = feat_c(end, :);
            c = c(1:end-1, :);
            feat_c = feat_c(1:end-1, :);
            
            
            cluster_map = zeros(size(sp));
            for j = 1:size(c, 1)
                list = find(c(j, :) == 1);
                for k = 1:length(list)
                    cluster_map(sp == list(k)) = j;
                end;
            end;
            
            imshow(mod(cluster_map, 255) + 1, cmap);
            drawnow;
%             F = getframe(gcf);
%             [x, Map ] = frame2im(F);
%             writeVideo(outputVideo, x)
            if (size(c, 1)  <20)
                pause;
            end;
        end;
%         close(outputVideo);
%         p = mdscale(dist, 2);
%         plot(p(:, 1), p(:, 2), '.');
%         d0 = sum((feat_a - feat_n).^2, 3);
%         d1 = sum((feat_a - feat_p).^2, 3);
%         hist(d0(:) - d1(:), -3:0.1:3);
    end;
    drawnow;
    idx = idx + 1;
end;
