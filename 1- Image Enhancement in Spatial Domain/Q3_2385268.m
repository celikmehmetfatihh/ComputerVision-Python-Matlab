% Simply I loaded the image, then I applied the max filter to enhance the
% image. First, for each pixel a 3x3 neighborhood for each pixel. I found
% the maximum pixel value, then i assigned it to Image3MaxOutput array. 
% Then I did the same operations just found the min pixel values for min
% filter operations to enhance the image. 
 
% Load the original image
Image3 = imread("Image3.png");

% Create a figure to show the original image
figure;
subplot(1, 2, 1);
imshow(Image3);
title("Original Image");

% Convert the image to double for convolution operations
Image3 = im2double(Image3);

% Get the dimensions of the original image
imgRow = size(Image3, 1);
imgColumn = size(Image3, 2);

% Create an empty array to hold the intermediate result
Image3MaxOutput = zeros(imgRow, imgColumn);

% Apply the max filter to enhance the image
for i = 2:imgRow - 1
    for j = 2:imgColumn - 1
        % Create a temporary array to hold a 3x3 neighborhood
        neighborhood = Image3(i-1 : i+1, j-1 : j+1);
        Image3MaxOutput(i, j) = max(neighborhood(:));  % Assign the maximum value to the output
    end
end

% Create an empty array to hold the final result
Image3Output = zeros(imgRow, imgColumn);

% Apply the min filter to remove noise
for i = 2:imgRow - 1
    for j = 2:imgColumn - 1
        % Create a temporary array to hold a 3x3 neighborhood
        neighborhood = Image3MaxOutput(i-1 : i+1, j-1 : j+1);
        Image3Output(i, j) = min(neighborhood(:));  % Assign the minimum value to the output
    end
end

subplot(1, 2, 2);
imshow(Image3Output);
title("Filtered Image");
