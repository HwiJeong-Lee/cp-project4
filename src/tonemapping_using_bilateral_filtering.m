function [ imgTM ] = tonemapping_using_bilateral_filtering( imgHDR, kernel_size, S, sigma1, sigma2, colorSpace )
    % parameters
    [height, width, channel] = size(imgHDR);
    N = height * width;
    e = 1e-6;

    % rgb colorspace
    if strcmp(colorSpace, 'rgb')
        imgTM = zeros(height, width, channel);
        
        for c = 1:channel
            L = log(imgHDR(:,:,c) + e);
            L_min = min(min(L));
            L_max = max(max(L));
            
            L_temp = (L - L_min) / (L_max - L_min);
            % bilateral filtering
            B_temp = bfilter2(L_temp, kernel_size, [sigma1 sigma2]);
            B = B_temp * (L_max - L_min) + L_min;
            D = L - B;
            B_ = S * (B - max(max(B)));
            I_TM = exp(B_ + D);
            
            imgTM(:,:,c) = I_TM;
        end
        
    % xyY colorspace
    elseif strcmp(colorSpace, 'xyY')
        img_XYZ = rgb2xyz(imgHDR, 'Colorspace', 'linear-rgb');

        X = img_XYZ(:,:,1);
        Y = img_XYZ(:,:,2);
        Z = img_XYZ(:,:,3);

        x = X ./ (X + Y + Z);
        y = Y ./ (X + Y + Z);

        L = log(Y + e);
        L_min = min(min(L));
        L_max = max(max(L));

        L_temp = (L - L_min) / (L_max - L_min);

        B_temp = bfilter2(L_temp, kernel_size, [sigma1 sigma2]);
        B = B_temp * (L_max - L_min) + L_min;
        D = L - B;
        B_ = S * (B - max(max(B)));
        I_TM = exp(B_ + D);

        [X, Y, Z] = xyY_to_XYZ(x, y, I_TM);

        img_XYZ_ = zeros(height, width, channel);
        img_XYZ_(:,:,1) = X;
        img_XYZ_(:,:,2) = Y;
        img_XYZ_(:,:,3) = Z;

        imgTM = xyz2rgb(img_XYZ_);

    end
end
