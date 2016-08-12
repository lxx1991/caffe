function edge_map = gen_edge_local_filler(label, filler_size)
    
    edge_map = zeros(size(label, 1), size(label, 2), filler_size * filler_size);
    i = (filler_size - 1) / 2;
    label = padarray(label, [i i], 0, 'both');
   
    for j = i + 1:size(label, 1)-i
        for k = i + 1:size(label, 2)-i
            for x = -i:i
                for y = -i:i
                    if (label(j, k) == 255 || label(j+x, k+y) == 255)
                        edge_map(j-i, k-i, (x+i) * filler_size + y + i + 1) = 1;
                    else
                        edge_map(j-i, k-i, (x+i) * filler_size + y + i + 1) = (label(j, k) == label(j+x, k+y)); 
                    end;
                end;
            end;
        end;
    end;
    edge_map = edge_map ./ repmat(sum(edge_map , 3), [1, 1, size(edge_map, 3)]);
end
