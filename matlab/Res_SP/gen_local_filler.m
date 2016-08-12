function output = gen_local_filler(sp, filler_size)

    output = zeros(size(sp, 1), size(sp, 2), filler_size * filler_size);
    sp_p = padarray(sp,[(filler_size - 1) / 2, (filler_size - 1) / 2], -1, 'both');
    
    cnt = 1;
    for i = 1:filler_size
        for j = 1:filler_size
            output(:, :, cnt) = single(sp_p(i:i+size(sp, 1)-1, j:j+size(sp, 2)-1) == sp);
            cnt = cnt + 1;
        end;
    end;
    output = output ./ repmat(sum(output , 3), [1, 1, cnt - 1]);
end