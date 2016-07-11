clear all; clc;

vlfeat_root = fullfile('/', 'DATA3', 'vlfeat-0.9.20', 'toolbox', 'vl_setup');
run(vlfeat_root);


%%
img_dir = fullfile('/DATA3/VOC_arg_instance/', 'img');
img_files = dir(img_dir);

REGIONSIZE = 20; 
REGULARIZER = 0.1;

for i = 3:length(img_files)
    disp(i)
    img_name = img_files(i).name;
    im = imread(fullfile(img_dir, img_name));
    [ segments, o ] = segment_slic(im, REGIONSIZE, REGULARIZER);
%    imshow(o);
%    title(max(segments(:)));
%    drawnow;
    break;
%     imwrite(segments, fullfile(output_dir, [img_name(1:end-4), '.png']));
    pause;
end;