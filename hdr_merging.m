function [ imgHDR ] = hdr_merging( imgs, g, exposures, weight, IMAGE_TYPE, MERGING_SCHEME )

    img_num = size(imgs, 2);
    [height, width, channel] = size(imgs{1});

    imgHDR = zeros(height, width, channel);

    for c = 1:channel
        for h = 1:height
            for w = 1:width
                num = 0;
                denom = 0;
                for k = 1:img_num
                    I_cur = imgs{k}(h, w, c);
                    w_cur = weight(I_cur + 1);
                    ln_t_cur = exposures(k);                    
                    
                    if strcmp(IMAGE_TYPE, 'rendered')
                        % convert non-linear image to linear
                        I_cur = exp(g(I_cur + 1));
                    elseif strcmp(IMAGE_TYPE, 'raw')
                        I_cur = I_cur / 255.;
                    end

                    denom = denom + w_cur;
                    
                    if strcmp(MERGING_SCHEME, 'linear')
                        num = num + w_cur * I_cur / exp(ln_t_cur);
                    elseif strcmp(MERGING_SCHEME, 'logarithmic')
                        num = num + w_cur * (log(I_cur) - ln_t_cur);
                    end                                     
                end
                imgHDR(h, w, c) = num / denom;                
            end
        end
    end
    
    if strcmp(MERGING_SCHEME, 'logarithmic')
        imgHDR = exp(imgHDR);
    end                                     

    % remove NAN or INF
    idx = find(isnan(imgHDR) | isinf(imgHDR));
    imgHDR(idx) = 0;    
end