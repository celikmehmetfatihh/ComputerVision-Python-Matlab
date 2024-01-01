% Simply I loaded the image, calculated the histogram of it. I chose a
% 170 as a threshold and did the operation. I set pixels with intensities 
% greater than or equal to the threshold (170) to 1, and those below the
% threshold to 0. Saved and displayed the threshold image. 


% Load the image
img = imread('Image1.png');

% Calculate the histogram of the image
histogramImg = zeros(256, 1);
[M, N] = size(img);

% Loop through each pixel to build the histogram
for i = 1:M
    for j = 1:N
        pixel = img(i, j);
        histogramImg(pixel + 1) = histogramImg(pixel + 1) + 1;
    end
end

% Find a threshold value based on the histogram. I chose 170 as a threshold
% value.
threshold = 170;

% Apply thresholding to create a binary image
binaryImg = img >= threshold;
img_output = uint8(binaryImg) * 255;

% Save the reconstructed image
imwrite(img_output, 'img1Output.png');

figure;

% Display the original image in the upper-left
subplot(2, 2, 1);
imshow(img);
title('Original Image');

% Display the histogram of the original image in the upper-right
subplot(2, 2, 2);
bar(histogramImg);
title('Histogram of Original Image');
xlabel('Pixel Value (0-255)');
ylabel('Frequency');

% Display the reconstructed image in the lower-left
subplot(2, 2, 3);
imshow(img_output);
title('Reconstructed Image');

% Calculate and plot the histogram of the reconstructed image in the lower-right
histogramImg_output = zeros(256, 1);
[M_output, N_output] = size(img_output);

% Loop through each pixel to build the histogram of the reconstructed image
for i = 1:M_output
    for j = 1:N_output
        pixel_output = img_output(i, j);
        histogramImg_output(pixel_output + 1) = histogramImg_output(pixel_output + 1) + 1;
    end
end

subplot(2, 2, 4);
bar(histogramImg_output);
title('Histogram of Reconstructed Image');
xlabel('Pixel Value (0-255)');
ylabel('Frequency');
