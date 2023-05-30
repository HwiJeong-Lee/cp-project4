% starter script for project 3
DO_HDR_IMAGING = true;
DO_EVALUATION = true;
DO_PHOTOGRAPHIC_TONEMAPPING = false;
DO_TONEMAPPING_USING_BILATERAL_FILTERING = false;

% parameters
NUM_IMAGES = 16;
IMAGE_TYPE = 'raw';
MERGING_SCHEME = 'logarithmic';
WEIGHT_SCHEME = 'uniform';

hdr_path = sprintf('../results/HDR images/HDR_%s_%s_%s.hdr', IMAGE_TYPE, MERGING_SCHEME, WEIGHT_SCHEME);

resize_ratio = 1/200;
l = 200;

% exposures tk
exposures = zeros(1, NUM_IMAGES);
for k = 1:NUM_IMAGES
    exposures(1, k) = log(power(2, k - 1) / 2048);
end

if DO_HDR_IMAGING

    % LINEARIZE RENDERED IMAGES (25 POINTS)
    imgs = {};
    imgs_small = {};

    % Read images and compute g function
    for i = 1:NUM_IMAGES
        img_path = sprintf('../data/door_exposure_stack/exposure_stack/exposure%d.jpg', i);
        img = imread(img_path);

        % Resize images for practical computation of g function
        img_small = imresize(img, resize_ratio, 'bilinear');
        [height, width, ch] = size(img_small);
        imgs_small{i} = double(img_small);
    end
   
    pixel_num = height * width
    pixel_indices = zeros(1, pixel_num);

    for i = 1:pixel_num
        pixel_indices(1, i) = i;
    end

    % Construct Z matrix for gsolve function
    Z = zeros(height*width, NUM_IMAGES, 3);
    for i = 1:NUM_IMAGES
        for c = 1:3
            Z(:, i, c) = imgs_small{i}(pixel_indices + (c - 1) * pixel_num);
        end
    end

    g_function = zeros(256, 3);
    for c = 1:3
        g_function(:, c) = gsolve(Z(:,:,c), t, l, w);
    end

    Zmin = ceil(0.01* 255);
    Zmax = floor(0.99 * 255);


    figure;
    plot(Zmin:Zmax, g_function(Zmin:Zmax, 1), 'r');
    hold on;
    plot(Zmin:Zmax, g_function(Zmin:Zmax, 2), 'g');
    hold on;
    plot(Zmin:Zmax, g_function(Zmin:Zmax, 3), 'b');
    hold off;
    title(sprintf('the function g: %s / %s / %s', IMAGE_TYPE, MERGING_SCHEME, WEIGHT_SCHEME));
    saveas(gcf, sprintf('../results/g plot/g plot_%s_%s_%s.png', IMAGE_TYPE, MERGING_SCHEME, WEIGHT_SCHEME))


    if strcmp(IMAGE_TYPE, 'raw')
        % Read images for raw type
        for i = 1:NUM_IMAGES
            img_path = sprintf('../data/tiff/exposure%d.tiff', i);
            img = imread(img_path);
            img = img(1:4000, 1:6000, :);
            img = imresize(img, 0.1);
            img = im2uint8(img);

            img(img < Zmin) = Zmin;
            img(img > Zmax) = Zmax;
            
            imgs{i} = double(img);
        end        
    elseif strcmp(IMAGE_TYPE, 'rendered')
        % Read images for rendered type
        for i = 1:NUM_IMAGES
            img_path = sprintf('../data/door_exposure_stack/exposure_stack/exposure%d.jpg', i);
            img = imread(img_path);
            img = imresize(img, 0.1);
            imgs{i} = double(img);
        end
    end
    
    % MERGE EXPOSURE STACK INTO HDR IMAGE (15 POINTS)
    
    % Weighting scheme (uniform / tent)
    w = weight_scheme(WEIGHT_SCHEME, Zmin, Zmax);

    hdr_img = hdr_merging(imgs, g_function, exposures, w, IMAGE_TYPE, MERGING_SCHEME);

    % Save the HDR image
    hdrwrite(hdr_img, hdr_path);
end

% EVALUATION (10 POINTS)
if DO_EVALUATION
    hdr_img = hdrread(hdr_path);
    
    % Display the HDR image
    %figure;
    %imshow(imgHDR);
    %title('Select neutral patches');

    % Define the number of neutral patches
    %num_patches = 6;
    
    % Initialize patch_coordinates array
    %patch_coordinates = zeros(num_patches, 4);

    % Get coordinates for each neutral patch
    %for i = 1:num_patches
        % Get user input for patch coordinates
        %fprintf('Select top-left and bottom-right corners of neutral patch %d\n', i);
        %fprintf('Click on the top-left corner of the patch.\n');
        %[x1, y1] = ginput(1);
        %fprintf('Click on the bottom-right corner of the patch.\n');
        %[x2, y2] = ginput(1);
        
        % Round the coordinates to integers
        %x1 = round(x1);
        %y1 = round(y1);
        %x2 = round(x2);
        %y2 = round(y2);
        
        % Store the coordinates in the patch_coordinates array
        %patch_coordinates(i, :) = [x1, y1, x2-x1, y2-y1];
        
        % Draw a rectangle to visualize the selected patch
        %hold on;
        %rectangle('Position', [x1, y1, x2-x1, y2-y1], 'EdgeColor', 'r', 'LineWidth', 2);
        %hold off;
    %end

    % Display the final patch coordinates
    %disp('Neutral patch coordinates:');
    %disp(patch_coordinates);

    img_XYZ = rgb2xyz(hdr_img, 'Colorspace', 'linear-rgb');
    
    L = img_XYZ(:,:,2);

    L_x = zeros(6, 1);
    L_y = zeros(6, 1);
    
    % coordinates of the cropped squares
    top_left = [375 62; 376 78; 376 93; 377 110; 377 125; 378 141;];
    bottome_right = [387 73; 388 89; 388 105; 388 122; 389 138; 390 153;];

    for i = 1:6       
        L_y(i) = mean(mean(L(top_left(i,2):bottome_right(i,2), top_left(i,1):bottome_right(i,1))));
        L_x(i) = i;
    end
    
    % Linear regression using fitlm
    lr = fitlm(L_x, L_y);
    % Compute the predicted values
    predicted_y = predict(lr, L_x);
    % Compute the least-squares error
    LSE = sum((L_y - predicted_y).^2);
    
    % Plot the regression results
    figure;
    plot(lr);
    title(sprintf('R-squared: %.3f, Least-Squares Error: %.3f', lr.Rsquared.Ordinary, LSE));
    xlabel('');
    ylabel('');
    saveas(gcf, sprintf('../results/regression plot/regression plot_%s_%s_%s.png', IMAGE_TYPE, MERGING_SCHEME, WEIGHT_SCHEME))
end

% PHOTOGRAPHIC TONEMAPPING (20 POINTS)
if DO_PHOTOGRAPHIC_TONEMAPPING
    % parameters of phtographic tonemapping
    K_rgb = 0.7;
    B_rgb = 0.9;
    K_xyY = 0.15;
    B_xyY = 0.95;
    hdr_img = double(hdrread(hdr_path));

    tonemapped_rgb = photographic_tonemapping(hdr_img, K_rgb, B_rgb, 'rgb');
    tonemapped_xyY = photographic_tonemapping(hdr_img, K_xyY, B_xyY, 'xyY');

    imwrite(tonemapped_rgb, '../results/photographic/rgb.png');    
    imwrite(tonemapped_xyY, '../results/photographic/xyY.png');    

    % Plot representative tonemaps
    figure;
    subplot(1, 2, 1);
    imshow(tonemapped_rgb);
    title('RGB tonemap');
    subplot(1, 2, 2);
    imshow(tonemapped_xyY);
    title('xyY tonemap');
    impixelinfo();
end

% TONEMAPPING USING BILATERAL FILTERING (30 POINTS)
if DO_TONEMAPPING_USING_BILATERAL_FILTERING
    % parameters of tonmapping useing bilateral filtering
    kernel_size = 5;
    S_rgb = 0.20;
    sigma_d = 2;
    sigma_r = 0.2;
    hdr_img = double(hdrread(hdr_path));

    tonemapped_rgb = tonemapping_using_bilateral_filtering(hdr_img, kernel_size, S_rgb, sigma_d, sigma_r, 'rgb');
    tonemapped_xyY = tonemapping_using_bilateral_filtering(hdr_img, kernel_size, S_rgb, sigma_d, sigma_r, 'xyY');

    imwrite(tonemapped_rgb, '../results/bilateral filtering/rgb.png');    
    imwrite(tonemapped_xyY, '../results/bilateral filtering/xyY.png');    

     figure;
     subplot(1, 2, 1);
     imshow(tonemapped_rgb);
     title('RGB tonemap');
     subplot(1, 2, 2);
     imshow(tonemapped_xyY);
     title('xyY tonemap');
end