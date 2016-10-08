function [ img, label ] = im_tf( img, label)
%IMAGE_TRANSFORME Summary of this function goes here
    %   Detailed explanation goes here
    
    if (rand()>0.5)
        img = img(:, end:-1:1, :);
        label = label(:, end:-1:1, :);
    end;

    img = imresize(img, [299, 299]);
end

