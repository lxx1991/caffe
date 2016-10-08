function [ img, label ] = im_tf( img, label)
%IMAGE_TRANSFORME Summary of this function goes here
    %   Detailed explanation goes here

    rescale_l = 0.8;
    rescale_h = 1.2;

    if (rand()>0.5)
        img = img(:, end:-1:1, :);
        label = label(:, end:-1:1, :);
    end;

    rescale = rescale_l + rand() * (rescale_h - rescale_l);
    img = imresize(img, rescale);
    label = imresize(label, rescale, 'nearest');
    
    height_img = size(img, 1);
    width_img = size(img, 2);
    
    th = floor(height_img / 32) * 32;
    tw = floor(width_img / 32) * 32;

    st_h = height_img - th;
    st_w = width_img - tw;
    

    tmp = randi(st_h + 1);
    img = img(tmp:tmp + th - 1, :, :);
    label = label(tmp:tmp + th - 1, :, :);

    tmp = randi(st_w + 1);
    img = img(:, tmp:tmp + tw - 1, :);
    label = label(:, tmp:tmp + tw - 1, :);

end

