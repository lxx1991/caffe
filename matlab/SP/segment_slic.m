function [ segments, o ] = segment_slic( im, feat, REGIONSIZE, REGULARIZER )  
    %UNTITLED3 Summary of this function goes here  
    %   Detailed explanation goes here  
    if (size(feat, 3) == 3)
        I =  vl_rgb2xyz(feat);  
        I_single = single(I);  
    else
        I_single = feat;
    end;
    segments = vl_slic(I_single, REGIONSIZE, REGULARIZER, 'MinRegionSize', 11);
    segments = uint16(segments);
    I = im;
    [m, n] = size(segments);  
    for i=1:m  
        for j = 1:n  
            label = segments(i,j);  
            labelTop = label;  
            if i>1  
                labelTop = segments(i-1,j);  
            end  
              
            labelBottom = label;  
            if i<m-1  
                labelBottom = segments(i+1,j);  
                 
            end  
            labelLeft = label;  
                if j > 1   
                    labelLeft = segments(i,j - 1);  
                end  
                  
                labelRight = label;  
                if j < n-1   
                    labelRight = segments(i,j + 1);  
                end  
            if label ~= labelTop || label ~= labelBottom || label ~= labelLeft || label ~= labelRight  
                I(i,j,1)=255;  
                I(i,j,2)=255;  
                I(i,j,3)=255;  
            end  
                  
        end  
          
        o = I;  
    end  
      
      
    end  