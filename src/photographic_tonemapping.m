function [ imgTMP ] = photographic_tonemapping( imgHDR, K, B, colorSpace )
    % parameters
    [height, width, channel] = size(imgHDR);
    N = height * width;
    e = 1e-6;
    
    % rgb colorspace
    if strcmp(colorSpace, 'rgb')
        imgTMP = zeros(height, width, channel);
        
        for c = 1:channel
            I_HDR = imgHDR(:,:,c);
            I_m_HDR = exp(mean(mean(log(I_HDR + e))));

            I_HDR_ = (K / I_m_HDR) * I_HDR;
            I_white_ = B * max(max(I_HDR_));

            I_HDR_tonemapping = I_HDR_ .* (1 + I_HDR_ / (I_white_)^2) ./ (1 + I_HDR_);

            imgTMP(:,:,c) = I_HDR_tonemapping;
        end
        
    % xyY colorspace
    elseif strcmp(colorSpace, 'xyY')
        img_XYZ = rgb2xyz(imgHDR, 'Colorspace', 'linear-rgb');

        X = img_XYZ(:,:,1);
        Y = img_XYZ(:,:,2);
        Z = img_XYZ(:,:,3);

        x = X ./ (X + Y + Z);
        y = Y ./ (X + Y + Z);

        I_m_HDR = exp(mean(mean(log(Y + e))));

        I_HDR_ = (K / I_m_HDR) * Y;
        I_white_ = B * max(max(I_HDR_));

        I_HDR_tonemapping = I_HDR_ .* (1 + I_HDR_ / (I_white_ * I_white_)) ./ (1 + I_HDR_);

        [X, Y, Z] = xyY_to_XYZ(x, y, I_HDR_tonemapping);

        img_XYZ_ = zeros(height, width, channel);
        img_XYZ_(:,:,1) = X;
        img_XYZ_(:,:,2) = Y;
        img_XYZ_(:,:,3) = Z;

        imgTMP = xyz2rgb(img_XYZ_);

    end
end
