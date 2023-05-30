function [ weight ] = weight_scheme( WEIGHT_TYPE, Zmin, Zmax )

    weight = zeros(256, 1);

    if strcmp(WEIGHT_TYPE, 'uniform')
        weight(Zmin:Zmax) = 1;
    elseif strcmp(WEIGHT_TYPE, 'tent')
        for i = Zmin:Zmax
            weight(i) = min(i - 1, 256 - i - 1) / 255.;
        end
    end
end













