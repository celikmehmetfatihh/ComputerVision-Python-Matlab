import cv2
import matplotlib.pyplot as plt
import os

# Setting the path to our dataset folder.
path = "C:\\Users\\M\\Desktop\\SPRING 2022-2023\\CNG 483\\Assignment-1\\Dataset"

def FeatureMatcher(keypointsList, descriptorsList):
    # Used algorithm is Fast Library for Approximate Nearest Neighbors (FLANN). Discovers a query point's closest neighbor in a big dataset.
    # This index will be created using k-d tree data structure
    FlannIndex = 1
    # I chose number of trees to use 6 because when I try more, the compile time would get more slower.
    indexes = dict(algorithm=FlannIndex, trees=6)
    # Checks value represents how many times the tree should be scanned in recursive search for the ideal match. I used 36, default value was 32, to get better results.
    searches = dict(checks=36)
    # Takes index and search parameters as a paramater. 
    flannMatched = cv2.FlannBasedMatcher(indexes, searches)
    
    # This list will store the matching key points and their respective distances.
    totalMatches  = []
    for i in range(len(keypointsList) - 1):
        keypoints = keypointsList[i] #  Current reference image's key points
        descriptors = descriptorsList[i]#  Current reference image's descriptors
        keypointsNext = keypointsList[i+1]#  Next reference image's key points
        descriptorsNext = descriptorsList[i+1]#  Next reference image's descriptors
        
        # gets current reference image's key points and descriptors and k is the amount to be returned for the nearest neighbors. So we'll get 2 of the most suitable matches for each descriptor
        # knnMatch will return k nearest matches between these two descriptors.
        #So, matchList will have list of list, each list having a match which is descriptor and the corresponding closest match which is descriptorsNext
        matchList = flannMatched.knnMatch(descriptors, descriptorsNext, k=2)
        
        # Each inner list of maskOfMatches is initialized with [0,0]. After that we'll use this list to remove any matches that fail Lowe's ratio test.
        maskofMatches = [[0, 0] for i in range(len(matchList))]
        # a and b will be the matched object.
        matches = []
        for i, (a, b) in enumerate(matchList):
            # According to Lowe's ratio test, a match is good a match if the distance between it and the closest match (a) is less than 0.7 times that of the second-closest match (b).
            if a.distance < 0.7 * b.distance:
                # describes that in that index match has been passed.
                maskofMatches[i] = [1, 0]
                # I learnt this from an online source that a.queryIdx and a.trainIdx are the indexes of the keypoints which corresponds to the current match.
                matches.append([keypoints[a.queryIdx], keypointsNext[a.trainIdx], a.distance])
        
        # Since the distance is the third element in our matches list, we sorted according to that.
        matches.sort(key=lambda x: x[2])
        # Delete other than top 10 matches as you declared in the assignment.
        matches = matches[:10]
        
         # Printing top 10 matching key points
        print(f"\nTop 10 matched keypoints between image {i+1} and image {i+2}:")
        for match in matches:
            # printing match[0].pt used for getting the coordinates, and first f makes {match[0].pt} a placeholder.
            print(f"Keypoint 1: {match[0].pt} matched with Keypoint 2:{match[1].pt} with distance {match[2]}")
        # Add the top ten matching key points
        totalMatches.append(matches)
    
    # Plotting
    for i in range(len(totalMatches)):
        # reads ith image and saves to currentImage
        currentImage = cv2.imread(os.path.join(path, f"frame{i}.jpg"))
        # reads ith+1 image and saves to nextImage
        nextImage = cv2.imread(os.path.join(path, f"frame{i+1}.jpg"))
        # Connects the keypoints which are matched between currentImage and nextImage
        drawMatches = cv2.drawMatches(currentImage, keypointsList[i], nextImage, keypointsList[i+1], totalMatches[i], None)
        plt.imshow(drawMatches)
        plt.show()
        
    

def FeatureExtractor():
    # Reading all the images in “Dataset” folder. 
    # os.listdir(path) returns all the directories and files from out Dataset.
    # os.path.join(path, f) joins the path with any f in the for loop, f stands for any directory or file from our path. The loop for retrieving all of the files and folders from our Dataset.
    images = [cv2.imread(os.path.join(path, f)) for f in os.listdir(path)]

    return images

def DetectSIFTFeatures():
    # We got the image list
    images = FeatureExtractor()
    # Creating our Scale-Invariant Feature Transform (SIFT) object.
    siftObj = cv2.SIFT_create()
    # Will represent a location that is distinctive, and likely to be matched with another images.
    keypointsList = []
    #  Will represent as a vectors representing the features of a keypoint. While a descriptor list consists of vectors that define the characteristics of each of these points, a keypoint list contains the positions of interesting points in an image.
    descriptorsList = []
    
    for image in images:
        # paramters: input image, optional mask parameter that is for describing the region of the image we want to detect. Since we use all of the image, we use None as a mask parameter. 
        keypoints, descriptors = siftObj.detectAndCompute(image, None)
        # Adding the keypoints and descriptors we detected to our lists.
        keypointsList.append(keypoints)
        descriptorsList.append(descriptors)
    
    return keypointsList, descriptorsList

def main():
    # Reading the reference images.
    # os.path.join finds the full path for the reference images.
    reference_1 = cv2.imread(os.path.join(path, "reference_1.jpg"))
    reference_2 = cv2.imread(os.path.join(path, "reference_2.jpg"))

    # Scale the reference images by a factor of 2 and 5
    # cv2.resize parameters:
    # source of time image, size of the output image since we did scaling we can say None, fx: horizontal scaling factor, fy: vertical scaling factor,
    #interpolation technic we used is Linear.
    transformed1_S2 = cv2.resize(reference_1, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    transformed1_S5 = cv2.resize(reference_1, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    transformed2_S2 = cv2.resize(reference_2, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    transformed2_S5 = cv2.resize(reference_2, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

    # Save the scaled images to our dataset folder.
    cv2.imwrite(os.path.join(path, "transformed1_S2.jpg"), transformed1_S2)
    cv2.imwrite(os.path.join(path, "transformed1_S5.jpg"), transformed1_S5)
    cv2.imwrite(os.path.join(path, "transformed2_S2.jpg"), transformed2_S2)
    cv2.imwrite(os.path.join(path, "transformed2_S5.jpg"), transformed2_S5)

    # Extracting the width and height of the image using shape attribute.
    (height, width) = reference_1.shape[:2]
    # Calculating the center point of the image. We used integer division. Later we'll use this center point to rotate the image.
    center = (int(width/2), int(height/2))
    scale = 1.0
    # 30 is the rotation angle, 1.0 is the scaling factor (since there is no scaling factor we used 1.0 as a default) 
    M1 = cv2.getRotationMatrix2D(center, 30, scale)
    M2 = cv2.getRotationMatrix2D(center, 150, scale)
    # paramters: image, transformation matrix which tells how the image should be transformed. , third argument: size of the image 
    transformed1_R30 = cv2.warpAffine(reference_1, M1, (width, height))
    transformed1_R130 = cv2.warpAffine(reference_1, M2, (width, height))

    # The same is applied for the reference_2 image.
    (height, width) = reference_2.shape[:2]
    center = (int(width/2), int(height/2))
    M1 = cv2.getRotationMatrix2D(center, 30, scale)
    M2 = cv2.getRotationMatrix2D(center, 150, scale)
    transformed2_R30 = cv2.warpAffine(reference_2, M1, (width, height))
    transformed2_R130 = cv2.warpAffine(reference_2, M2, (width, height))

    # Save the rotated images to our dataset folder.
    cv2.imwrite(os.path.join(path, "transformed1_R30.jpg"), transformed1_R30)
    cv2.imwrite(os.path.join(path, "transformed1_R130.jpg"), transformed1_R130)
    cv2.imwrite(os.path.join(path, "transformed2_R30.jpg"), transformed2_R30)
    cv2.imwrite(os.path.join(path, "transformed2_R130.jpg"), transformed2_R130)
    
    keypointsList, descriptorsList = DetectSIFTFeatures() # calls FeatureExtractor() inside
    FeatureMatcher(keypointsList, descriptorsList)
    
    """(8 points) Discuss the trend in the plotted results. What is the effect of increasing the scale
on the matching distance? 
            The matching distance will grow as an image's scale increases. While image's scale increases, the descriptors of its
        keypoints increases, which results wider gaps between matches. Large descriptors will make feature spaces larger.
    """
    
    """"
    (8 points) Discuss the trend in the plotted results. What is the effect of the angle of rotation
on the matching distance? 
            The matching distance grows as the rotational angle grows. While we have a low matching distance, the angle of rotation is small because the features of the two images may be easily matched. The features 
        in the two images differ more as the angle of rotation grows, making it more difficult for the algorithm to identify matches.
    """