addpath(fullfile('..', '..', 'data', 'VOCcode'));
VOCinit;
class_hash = containers.Map();
for i = 1:VOCopts.nclasses
    class_hash(VOCopts.classes{i}) = i;
end;

dir_dataset = fullfile('..', '..', 'data', 'VOC_arg');
train_list = fullfile(dir_dataset, 'train1.txt');

[name_list, label_list, useless]= textread(train_list, '%s %s %s');

object = cell(1, 20);

for idx = 1:length(name_list)
    disp(idx);
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
            subplot(211);
            imshow(patch, cmap);
            subplot(212);
            imshow(img(bb(2):bb(4), bb(1):bb(3), :));
            drawnow;
        end;
    end;
end;
