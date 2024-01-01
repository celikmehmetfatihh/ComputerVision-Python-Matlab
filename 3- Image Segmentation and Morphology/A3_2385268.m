inputImage = imread("image6.png");

segmentationOut = segmentation(inputImage);

calculatedNumbers = countNumber(segmentationOut);

function segmentationOut = segmentation(inputImage)
    % Grayscale conversion
    grayImage = im2gray(inputImage);
    figure
    
    % Displaying the grayscale version of the image
    subplot(1,2,1);
    imshow(grayImage);
    title('Grayscaled Image')
    
    % Grayscale to binary image
    binaryImage = im2bw(grayImage);
 
    % Display binary image
    subplot(1,2,2);
    imshow(binaryImage);
    title('Binary Image');
    segmentationOut = binaryImage;
end

function calculatedNumbers = countNumber(segmentationOut)
    % We know that the die are squares. If we take the root of the area,
    % we get the length of one side of it. We can use this information to
    % extract the die from the segmented image.
    rp = regionprops(segmentationOut);
    dieInfo = [rp.Area] > 5000;
    rp = rp(dieInfo);
    
    figure
    % I assume that there will be 2 die in the image for all cases
    for k=1:2
        diceLength = round(sqrt(rp(k).Area));
        xStartPos = round(rp(k).Centroid(1) - diceLength/2);
        xEndPos = round(rp(k).Centroid(1) + diceLength/2);
        yStartPos = round(rp(k).Centroid(2) - diceLength/2);
        yEndPos = round(rp(k).Centroid(2) + diceLength/2);
        
        dice = segmentationOut(yStartPos:yEndPos,xStartPos:xEndPos);
        
        %The lines below are specifically for cleaning out the dice image
        %So that only the dots on the dice are remaining.
        dice = DilateImage(dice);
        dice = ErodeImage(dice);
        dice = imcomplement(dice);
        dice = imclearborder(dice);
        calculatedNumbers(k) = numel(regionprops(dice));

        subplot(1,2,k)
        imshow(dice);
        title(calculatedNumbers(k));
    end 
end

function erodedImage = ErodeImage(binaryImage)
    se = strel('disk', 5);
    erodedImage = imerode(binaryImage, se);
end

function dilatedImage = DilateImage(binaryImage)
    se = strel('disk', 5);
    dilatedImage = imdilate(binaryImage, se);
end
