files = dir('/DATA3/VOC_arg_instance/img/');

for i = 1:100
    p = randi(length(files)-2) +2;

    im = imread(['/DATA3/VOC_arg_instance/img/', files(p).name(1:end-4), '.jpg']);
    sp = imread(['/DATA3/VOC_arg_instance/superpixel_20_0.1/', files(p).name(1:end-4), '.png']);
    
    step = 3;%randi(5);
    
    edge = sp_graph(double(sp), step);
    
    c = randi(max(sp(:) + 1)) - 1;
    
    subplot(121);
    imshow(im);
    
    [p, ~] = find(edge == c);
    for j = 1:length(p)
        if edge(p(j), 1) == c
            tmp = edge(p(j), 2);
        else
            tmp = edge(p(j), 1);
        end;
        im(repmat(sp == tmp, [1, 1, 3])) = 255;
    end;
    subplot(122);
    imshow(im);
    title(step);
    drawnow;
    pause;
end;