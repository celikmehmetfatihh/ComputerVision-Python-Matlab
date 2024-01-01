% --------------------------First Image--------------------------
% Load the first image
imgWithNoise = imread('noisy1.png');

% Define the filter dimensions
filterSize = 3;

imgRows = size(imgWithNoise, 1);
imgCols = size(imgWithNoise, 2);
midPointFilteredImg = zeros(imgRows, imgCols);

% Perform midfiltering on the image
for x = 2:(imgRows - 1)
    for y = 2:(imgCols - 1)
        % Get the surrounding pixels
        neighborhood = double(imgWithNoise(x-1:x+1, y-1:y+1));
        
        % Initialize variables to store the min and max values
        localMin = neighborhood(1, 1);
        localMax = neighborhood(1, 1);
        

        % Determine the min and max values
        for p = 1:filterSize 
            for q = 1:filterSize 
                localMin = min(neighborhood(p, q), localMin); 
                localMax = max(neighborhood(p, q), localMax); 
            end
        end
        
        % Calculate the average
        localMid = (localMax + localMin) / 2; 
        
        % Do the update
        midPointFilteredImg(x, y) = localMid; 
    end
end

% Define Sobel filters for edge detection
Hx = [-1 -2 -1; 0 0 0; 1 2 1];
Hy = [-1 0 1; -2 0 2; -1 0 1];

hxRow = size(Hx, 1);
hxColumn = size(Hx, 2);
hyRow = size(Hy, 1);
hyColumn = size(Hy, 2);

% Create arrays to hold the derivative of X and Y for edge detection
derivativeOfX = zeros(imgRows, imgCols);
derivativeOfY = zeros(imgRows, imgCols);

% Find horizontal edges using convolution on the processed image
for i = 2:(imgRows - 1)
    for j = 2:(imgCols - 1)
        sum = 0;
        for k = 1:hxRow
            for l = 1:hxColumn
                sum = sum + (Hx(k, l) * double(midPointFilteredImg(i + k - 2, j + l - 2)));
            end
        end
        derivativeOfX(i, j) = sum;
    end
end

% Find vertical edges using convolution on the processed image
for i = 2:(imgRows - 1)
    for j = 2:(imgCols - 1)
        sum = 0;
        for k = 1:hyRow
            for l = 1:hyColumn
                sum = sum + (Hy(k, l) * double(midPointFilteredImg(i + k - 2, j + l - 2)));
            end
        end
        derivativeOfY(i, j) = sum;
    end
end

% Calculate the absolute values of the derivatives
derivativeOfX = abs(derivativeOfX);
derivativeOfY = abs(derivativeOfY);

% Combine the absolute derivatives to find overall edges of the image
midPointEdges = sqrt(derivativeOfX.^2 + derivativeOfY.^2);


% Find horizontal edges using convolution on imgWithNoise
for i = 2:(imgRows - 1)
    for j = 2:(imgCols - 1)
        sum = 0;
        for k = 1:hxRow
            for l = 1:hxColumn
                sum = sum + (Hx(k, l) * double(imgWithNoise(i + k - 2, j + l - 2)));
            end
        end
        derivativeOfX(i, j) = sum;
    end
end

% Find vertical edges using convolution on imgWithNoise
for i = 2:(imgRows - 1)
    for j = 2:(imgCols - 1)
        sum = 0;
        for k = 1:hyRow
            for l = 1:hyColumn
                sum = sum + (Hy(k, l) * double(imgWithNoise(i + k - 2, j + l - 2)));
            end
        end
        derivativeOfY(i, j) = sum;
    end
end

% Calculate the absolute values of the derivatives
derivativeOfX = abs(derivativeOfX);
derivativeOfY = abs(derivativeOfY);

% Combine the absolute derivatives to find overall edges of the image
noisyEdges = sqrt(derivativeOfX.^2 + derivativeOfY.^2);

% Image to show noise reduction effectiveness
edgeDifference = noisyEdges - midPointEdges;

% Displaying the images
figure;
subplot(2, 4, 1), imshow(imgWithNoise), title('Original Noisy Image');
subplot(2, 4, 4), imshow(uint8(midPointFilteredImg)), title('Mid-Point Filtered');
subplot(2, 4, 5), imshow(uint8(noisyEdges)), title('Original Edges');
subplot(2, 4, 6), imshow(uint8(midPointEdges)), title('Mid-Point Filtered Edges');
subplot(2, 4, 8), imshow(uint8(edgeDifference)), title('Noise Reduction');


% --------------------------Second Image--------------------------
% Read the second image
noisy2image = imread('noisy2.png'); 

% Get the size of the second image
[M, N] = size(noisy2image); 

% Define the D0 which controls the spread of the notch
fsize = 70; 

% Create the filter for frequency domain processing using two loops
for u = 1:M
    for v = 1:N
        % Calculate the distance Dkp using the formula   
        Dkp(u, v) = sqrt((u - (M / 2) + 300)^2 + (v - (N / 2) + 0)^2); % Find D2, represents distance from specific points in the frequency domain
        % Calculate the distance Dkn using the formula 
        Dkn(u, v) = sqrt((u - (M / 2) - 300)^2 + (v - (N / 2) - 0)^2); % Find D1
    end
end

% Compute the Fourier Transform   
fourierTransform = fft2(noisy2image);

% Shift the zero-frequency component to the center  
centeredFourierTransform = fftshift(fourierTransform);

% Normalize the centered Fourier Transform 
normalizedFourierTransform = centeredFourierTransform / (M * N); 

F = normalizedFourierTransform;

% Gaussian band raject filter part
% Calculate the squared distances for Dkp and Dkn
squaredDkp = Dkp.^2;
squaredDkn = Dkn.^2;

% Finde the denominator value
denominatorValue = 2 * fsize^2;

% Perform the division operation for the exponential function for high-pass filters
dividedDkp = squaredDkp ./ denominatorValue;
dividedDkn = squaredDkn ./ denominatorValue;

% Calculate the negative exponential part for both high-pass filters
negativeExponentialDkp = exp(-dividedDkp);
negativeExponentialDkn = exp(-dividedDkn);

% Subtract 1 from the negative exponential results to get the high-pass
% filter components- notch pass = 1- notchreject, we finf notchreject
highPassFilterComponentDkp = negativeExponentialDkp - 1;
highPassFilterComponentDkn = negativeExponentialDkn - 1;

% Assign the high-pass filter components to Hp and Hn
Hp = highPassFilterComponentDkp; % High-pass filter centered at (uk, vk)
Hn = highPassFilterComponentDkn; % High-pass filter centered at (-uk, -vk)

% Multiply the two high-pass filter components to create a band-reject filter
Hnr = Hp .* Hn; % This multiplication results in the band-reject filter to create band reject filter

% Converting back to spatial domain
%Step1 Apply the band-reject filter to the frequency domain representation of the image
filteredFrequencyDomain = F .* Hnr;

%Step2
G = filteredFrequencyDomain;

%ifft2 3-6
% Step 3: Perform the inverse 2D Fourier transform
inverseFourierTransform = ifft2(G);

% Step 4: Assign the result of the inverse Fourier transform
inverseTransform = inverseFourierTransform;

% Step 5: Calculate the magnitude
magnitudeOfInverseTransform = abs(inverseTransform);

% Step 6: Assign the magnitude
filteredImage = magnitudeOfInverseTransform;

% Convert the filtered image to uint8 data type after normalization
% Initialize minValue and maxValue with the first element of the filteredImage
minValue = filteredImage(1);
maxValue = filteredImage(1);

% Get the number of rows and columns in the filteredImage - normalize 0,1,
% to grayscale
[rows, columns] = size(filteredImage);

% mat2gray instead of normalized value = (value - min value= / (max - min)

% Iterate over each element of the filteredImage using nested for loops
for i = 1:rows
    for j = 1:columns
        % Check if the current element is less than the current minValue
        if filteredImage(i, j) < minValue
            minValue = filteredImage(i, j);
        end
        
        % Check if the current element is greater than the current maxValue
        if filteredImage(i, j) > maxValue
            maxValue = filteredImage(i, j);
        end
    end
end


% Get the size of the filtered image
[rows, columns] = size(filteredImage);

% Initialize the normalizedImage with zeros of the same size as filteredImage
normalizedImage = zeros(rows, columns);

% Loop through every pixel in the filteredImage to apply normalization
for i = 1:rows
    for j = 1:columns
        % Apply the normalization formula to each pixel
        normalizedImage(i, j) = (filteredImage(i, j) - minValue) / (maxValue - minValue);
    end
end

% Step 7: Scale the normalized image
scaledImage = 255 * normalizedImage;

% Step 8: Convert the scaled image
filteredImage = uint8(scaledImage);
figure;
imshow(filteredImage);


% Define Sobel filters for edge detection
Hx = [-1 -2 -1; 0 0 0; 1 2 1];
Hy = [-1 0 1; -2 0 2; -1 0 1];
hxRow = size(Hx, 1);
hxColumn = size(Hx, 2);
hyRow = size(Hy, 1);
hyColumn = size(Hy, 2);


% Create arrays to hold the derivative
derivativeOfXOriginal = zeros(M, N);
derivativeOfYOriginal = zeros(M, N);

derivativeOfXFiltered = zeros(M, N);
derivativeOfYFiltered = zeros(M, N);

% Find horizontal edges
for i = 2:M-1
    for j = 2:N-1
        sum = 0;
        for k = 1:hxRow
            for l = 1:hxColumn
                sum = sum + (Hx(k, l) * noisy2image(i + k - 2, j + l - 2));
            end
        end
        derivativeOfXOriginal(i, j) = sum;
    end
end

% Find vertical edges
for i = 2:M-1
    for j = 2:N-1
        sum = 0; 
        for k = 1:hyRow 
            for l = 1:hyColumn 
                sum = sum + (Hy(k, l) * noisy2image(i + k - 2, j + l - 2)); 
            end
        end
        derivativeOfYOriginal(i, j) = sum;
    end
end

% Find horizontal edges
for i = 2:M-1
    for j = 2:N-1
        sum = 0;
        for k = 1:hxRow
            for l = 1:hxColumn 
                sum = sum + (Hx(k, l) * filteredImage(i + k - 2, j + l - 2));
            end 
        end
        derivativeOfXFiltered(i, j) = sum;
    end
end

% Find vertical edges using convolution on the filtered image
for i = 2:M-1
    for j = 2:N-1
        sum = 0;
        for k = 1:hyRow
            for l = 1:hyColumn
                sum = sum + (Hy(k, l) * filteredImage(i + k - 2, j + l - 2));
            end
        end
        derivativeOfYFiltered(i, j) = sum;
    end
end

% Calculate the absolute values of the derivatives for both original and filtered images
derivativeOfXOriginal = abs(derivativeOfXOriginal);
derivativeOfYOriginal = abs(derivativeOfYOriginal);
derivativeOfXFiltered = abs(derivativeOfXFiltered);
derivativeOfYFiltered = abs(derivativeOfYFiltered);

% Combine the absolute derivatives to find overall edges of the images
edgeOfnoisy2image = abs(derivativeOfXOriginal + derivativeOfYOriginal);
edgeOfFilteredImage = abs(derivativeOfXFiltered + derivativeOfYFiltered);
edgeofthefinal = edgeOfnoisy2image - edgeOfFilteredImage;

% Display the noise image
figure;
imshow(edgeofthefinal, []);
title('Noises Of the Final Image');

% --------------------------Third Image--------------------------
% Read noisy image 3
input_image = imread('noisy3.png'); 

% Find size of nosiy image 3
[rows, cols] = size(input_image); 

% Find the shifted DFT 
F = fft2( double(input_image) ) / (rows * cols);  
F = fftshift(F); 

% Butterworth filter constants definition
Dkp = zeros(rows, cols);    
Dkn = zeros(rows, cols);    

% For the Butterworth notchreject filter find the distances
for u=1:rows
    for v=1:cols
        Dkp(u,v) =  sqrt((u - (rows / 2) - 35)^2 + (v - (cols/2)  +33) ^ 2); % D1
        Dkn(u,v) =  sqrt((u - (rows / 2) + 35)^2 + (v - (cols/2) -33) ^ 2); % D2
    end
end


% Find the Butterworth Notch Reject Filter
Hnr = 1 ./ (1 + (35 ^ 2 ./ (Dkp .* Dkn)).^2 );

% Shift the image
ShiftedF = F .* Hnr;

% Compute the inverse DFT  
inversed = ifft2(fftshift(ShiftedF));

% Compute the absolute values
absoluted = abs(inversed);

% Normalize the image
absoluted = abs(inversed);

min_val = absoluted(1);
max_val = absoluted(1);

% Find out the minimum and maximum
for i = 1: numel(absoluted)
    if absoluted(i) < min_val
        min_val = absoluted(i);
    elseif absoluted(i) > max_val
        max_val = absoluted(i);
    end
end

% mat2gray
% Normalize the values using the calculated min and max
normalized_inversed = (absoluted - min_val) / (max_val - min_val);

% Scale the normalized values to the range [0, 255] and convert to uint8
filteredImage = uint8(255 * normalized_inversed);

% Display the filtered image
figure;
imshow(filteredImage);


% Define Sobel filters for edge detection
Hx = [-1 -2 -1; 0 0 0; 1 2 1];
Hy = [-1 0 1; -2 0 2; -1 0 1];
hxRow = size(Hx, 1);
hxColumn = size(Hx, 2);
hyRow = size(Hy, 1);
hyColumn = size(Hy, 2);

% Create arrays to hold the derivative of X and Y for edge detection
derivativeOfXFiltered = zeros(rows, cols);
derivativeOfYFiltered = zeros(rows, cols);

% Find horizontal edges using convolution on the filtered image
for i = 2:rows-1
    for j = 2:cols-1
        sumX = 0;
        sumY = 0;
        for k = 1:hxRow
            for l = 1:hxColumn
                sumX = sumX + (Hx(k, l) * filteredImage(i + k - 2, j + l - 2));
                sumY = sumY + (Hy(k, l) * filteredImage(i + k - 2, j + l - 2));
            end
        end
        derivativeOfXFiltered(i, j) = sumX;
        derivativeOfYFiltered(i, j) = sumY;
    end
end

% Find absolute values
derivativeOfXFiltered = abs(derivativeOfXFiltered);
derivativeOfYFiltered = abs(derivativeOfYFiltered);

edgeOfFilteredImage = abs(derivativeOfXFiltered + derivativeOfYFiltered);


subplot(1, 3, 1);
imshow(input_image, []);
title('Input Image');


subplot(1, 3, 2);
imshow(filteredImage, []);
title('Filtered Image');

subplot(1, 3, 3);
imshow(edgeOfFilteredImage, []);
title('Edges of the Filtered Image');