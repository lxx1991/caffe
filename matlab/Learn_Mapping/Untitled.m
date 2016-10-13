fid = fopen('/DATA3/caffe/examples/IRes_VOC_Mapping/model_r_pre.prototxt', 'r');
fid2 = fopen('/DATA3/caffe/examples/IRes_VOC_Mapping/model_r.prototxt', 'w');
cnt =  0;
while 1
    tline = fgetl(fid);
    
    if ~ischar(tline), break, end
    if strcmp(tline, '    name: "lxx_label"')
        tline = sprintf('    name: "%d"', cnt);
        cnt = cnt + 1;
    end;
    fprintf(fid2, '%s\n', tline);
end
fclose(fid);
fclose(fid2);