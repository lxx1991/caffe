function [ img1, img2, img3, img4, label ] = im_tf2( img1, img2, img3, img4, label)
%IMAGE_TRANSFORME Summary of this function goes here
    %   Detailed explanation goes here

    img = cat(3, img1, img2, img3, img4);
    
    rescale_l = 1.25;
    rescale_h = 1.25;

    rescale = rescale_l + rand() * (rescale_h - rescale_l);
    img = imresize(img, rescale);
    label = imresize(label, rescale, 'nearest');
    
    height_img = size(img, 1);
    width_img = size(img, 2);
    
    th = 320;
    tw = 320;

    st_h = height_img - th;
    st_w = width_img - tw;
    

    tmp = randi(st_h + 1);
    img = img(tmp:tmp + th - 1, :, :);
    label = label(tmp:tmp + th - 1, :, :);

    tmp = randi(st_w + 1);
    img = img(:, tmp:tmp + tw - 1, :);
    label = label(:, tmp:tmp + tw - 1, :);
    
    img1 = img(:, :, 1:3);
    img2 = img(:, :, 4:6);
    img3 = img(:, :, 7:9);
    img4 = img(:, :, 10:12);
end

