% Simply I loaded the image, prepared my filter 3x3, then I applied median
% filtering. First, I created a 3x3 neighborhood for each pixel, then
% sorted the pixel values to find the median.
% Then I did the edge detection. I defined two sobel filters as Hx, Hy. I
% performed the convolution to to detect horizontal and vertical edges in
% the original image. Then, I calculated the absolute values of these derivatives.
% Finally, I combined the absolute derivatives to find overall edges in the image.

% Load the image
Image2 = imread("Image2.png");
figure;
subplot(2, 2, 1);
imshow(Image2);
title("Original Image");

% Convert the image to double for convolution operations
Image2 = im2double(Image2);

% Define the size of the median filter
rowFilter = 3;
columnFilter = 3;

% Find the size of the image
imgRow = size(Image2, 1);
imgColumn = size(Image2, 2);

% Create an empty array to hold the final filtered image
Image2Output = zeros(imgRow, imgColumn);

% Apply median filtering to remove noise
for i = 2:imgRow-1
    for j = 2:imgColumn-1
        temp = zeros(1, 9);
        index = 1;

        % Traverse the 3x3 neighborhood
        for k = -1:1
            for l = -1:1
                temp(index) = Image2(i + k, j + l);
                index = index + 1;
            end
        end

        % Sort the pixel values in ascending order
        temp = sort(temp, "ascend");

        % Find the median value and store it in the filtered image
        Image2Output(i, j) = median(temp);
    end
end

% Display the enhanced image
subplot(2, 2, 2);
imshow(Image2Output);
title("Filtered Image");

% Define Sobel filters for edge detection
Hx = [-1 -2 -1; 0 0 0; 1 2 1];
Hy = [-1 0 1; -2 0 2; -1 0 1];

hxRow = size(Hx, 1);
hxColumn = size(Hx, 2);
hyRow = size(Hy, 1);
hyColumn = size(Hy, 2);

% Create arrays to hold the derivative of X and Y for edge detection
derivativeOfX = zeros(imgRow, imgColumn);
derivativeOfY = zeros(imgRow, imgColumn);

% Find horizontal edges using convolution on the original image
for i = 2:imgRow-1
    for j = 2:imgColumn-1
        sum = 0;
        for k = 1:hxRow
            for l = 1:hxColumn
                sum = sum + (Hx(k, l) * Image2(i + k - 2, j + l - 2));
            end
        end
        derivativeOfX(i, j) = sum;
    end
end

% Find vertical edges using convolution on the original image
for i = 2:imgRow-1
    for j = 2:imgColumn-1
        sum = 0;
        for k = 1:hyRow
            for l = 1:hyColumn
                sum = sum + (Hy(k, l) * Image2(i + k - 2, j + l - 2));
            end
        end
        derivativeOfY(i, j) = sum;
    end
end

% Calculate the absolute values of the derivatives
derivativeOfX = abs(derivativeOfX);
derivativeOfY = abs(derivativeOfY);

% Combine the absolute derivatives to find overall edges of the image
edgeOfImage2 = abs(derivativeOfX + derivativeOfY);

% Display the edges of the image
subplot(2, 2, 3);
imshow(edgeOfImage2);
title("Original Image Edges");

% Do the same thing for imgoutput
derivativeOfX = zeros(imgRow, imgColumn);
derivativeOfY = zeros(imgRow, imgColumn);

for i = 2:imgRow-1
    for j = 2:imgColumn-1
        sum = 0;
        for k = 1:hxRow
            for l = 1:hxColumn
                sum = sum + (Hx(k, l) * Image2Output(i + k - 2, j + l - 2));
            end
        end
        derivativeOfX(i, j) = sum;
    end
end

for i = 2:imgRow-1
    for j = 2:imgColumn-1
        sum = 0;
        for k = 1:hyRow
            for l = 1:hyColumn
                sum = sum + (Hy(k, l) * Image2Output(i + k - 2, j + l - 2));
            end
        end
        derivativeOfY(i, j) = sum;
    end
end

derivativeOfX = abs(derivativeOfX);
derivativeOfY = abs(derivativeOfY);

edgeOfImage2Output = abs(derivativeOfX + derivativeOfY);

% Display the edges of Image2Output
subplot(2, 2, 4);
imshow(edgeOfImage2Output);
title("Filtered Image Edges");
