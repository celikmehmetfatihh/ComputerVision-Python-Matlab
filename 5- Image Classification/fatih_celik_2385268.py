import os
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

"""
First, I choose mean and standard deviation features to use in the mystery_features function because both features get information about total contrast and brightness of the images in the dataset. Also they provide information about the distribution of the pixel values in the images.          How does it help to get good classification accuracy?
To give an example, if we have a dataset consist of images of fruits, we can distinguish ripe and unripe ones if we use mean and standard deviation as our features because ripe fruits generally has lower standard deviation than unripe fruits.

Then, I got bad accuricies, after that I changed myster_features to color information.
"""


def mystery_features(images):
    features = []
    for image in images:
        # color information
        color = np.mean(image, axis=(0, 1))
        features.append(color)
    return features


def hist_features(images):
    features = []
    for image in images:
        # bins default 256, 8-bit grayscale
        hist, bins = np.histogram(image, bins=256, range=(0, 256))
        features.append(hist)
    return features


def load_dataset(path):
    trainingImg = []
    trainingLabels = []
    validationImg = []
    validationLabels = []
    testingImg = []
    testingLabels = []

    # Take all the images formatted with .jpg(not the directories) and save it to images
    images = [
        file
        for file in os.scandir(path)
        if file.is_file() and file.name.endswith(".jpg")
    ]
    imagesTotal = len(images)
    trainingNum = int(imagesTotal * 0.5)
    validationNum = int(imagesTotal * 0.25)

    # Iterate from index 0 to trainingNum(%50 of the image)
    for file in images[:trainingNum]:
        imagePath = file.path
        image = Image.open(imagePath)
        # Convert to grayscale
        grayscaleImage = image.convert("L")
        trainingImg.append(grayscaleImage)

        # Extract class label from file name exluding numbers
        label = file.name.split(".")[0].rstrip("0123456789")
        trainingLabels.append(label)

    # Iterate from index trainingNum(%50) to trainingNum + validationNum(%75)
    for file in images[trainingNum : trainingNum + validationNum]:
        imagePath = file.path
        image = Image.open(imagePath)
        # Convert to grayscale
        grayscaleImage = image.convert("L")
        validationImg.append(grayscaleImage)

        # Extract class label from file name exluding numbers
        label = file.name.split(".")[0].rstrip("0123456789")
        validationLabels.append(label)

    # Iterte from index trainingNum + validationNum (%75) to %100 of the dataset
    for file in images[trainingNum + validationNum :]:
        imagePath = file.path
        image = Image.open(imagePath)
        # Convert to grayscale
        grayscaleImage = image.convert("L")
        testingImg.append(grayscaleImage)

        # Extract class label from file name exluding numbers
        label = file.name.split(".")[0].rstrip("0123456789")
        testingLabels.append(label)

    return (
        (trainingImg, trainingLabels),
        (validationImg, validationLabels),
        (testingImg, testingLabels),
    )


def training(train_features, train_labels, val_features, val_labels):
    k_values = [1, 3, 5, 7]
    accuracies = []

    train_features = np.array(train_features)
    n_samples = train_features.shape[
        0
    ]  # getting number of samples in the training data
    n_features = (
        train_features.size // n_samples
    )  # getting number of features in the training data
    train_features = train_features.reshape(
        n_samples, n_features
    )  # reshaping n_samples rows x n_features columns

    val_features = np.array(val_features)
    n_samples = val_features.shape[
        0
    ]  # getting number of samples in the validation data
    n_features = (
        val_features.size // n_samples
    )  # getting number of features in the validation data
    val_features = val_features.reshape(
        n_samples, n_features
    )  # reshaping n_samples rows x n_features columns

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_features, train_labels)

        accuracy = knn.score(val_features, val_labels)
        accuracies.append(accuracy)
        print(f"Accuracy for K={k}: {accuracy}")

    plt.plot(k_values, accuracies)
    plt.xlabel("K")
    plt.ylabel("Validation Accuracy")
    plt.show()

    bestK = int(input("Enter the best value of K: "))
    return bestK


def testing(best_k, test_features, test_labels, train_features, train_labels):
    train_features = np.array(train_features)
    n_samples = train_features.shape[
        0
    ]  # getting number of samples in the training data
    n_features = (
        train_features.size // n_samples
    )  # getting number of features in the training data
    train_features = train_features.reshape(
        n_samples, n_features
    )  # reshaping n_samples rows x n_features columns

    test_features = np.array(test_features)
    n_samples = test_features.shape[0]  # getting number of samples in the test data
    n_features = (
        test_features.size // n_samples
    )  # getting number of features in the test data
    test_features = test_features.reshape(
        n_samples, n_features
    )  # reshaping n_samples rows x n_features columns

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_features, train_labels)
    accuracy = knn.score(test_features, test_labels)
    return accuracy


if __name__ == "__main__":
    path = "C:\\Users\\M\\Desktop\\SPRING 2022-2023\\CNG 483\\Assignment-2\\Dataset"
    (
        (trainingImg, trainingLabels),
        (validationImg, validationLabels),
        (testingImg, testingLabels),
    ) = load_dataset(path)

    print(f"Total training images: {len(trainingImg)}")
    print(f"Total training labels: {len(trainingLabels)}")
    print(f"Total validation images: {len(validationImg)}")
    print(f"Total validation labels: {len(validationLabels)}")
    print(f"Total testing images: {len(testingImg)}")
    print(f"Total testing labels: {len(testingLabels)}")

    histFeaturesTraining = hist_features(trainingImg)
    histFeaturesValidation = hist_features(validationImg)
    histFeaturesTesting = hist_features(testingImg)
    print(f"Total hist_features Training: {len(histFeaturesTraining)}")
    print(f"Total hist_features Validation: {len(histFeaturesValidation)}")
    print(f"Total hist_features Testing: {len(histFeaturesTesting)}")

    mysteryFeaturesTraining = mystery_features(trainingImg)
    mysteryFeaturesValidation = mystery_features(validationImg)
    mysteryFeaturesTesting = mystery_features(testingImg)
    print(f"Total mystery_features Training: {len(mysteryFeaturesTraining)}")
    print(f"Total mystery_features Validation: {len(mysteryFeaturesValidation)}")
    print(f"Total mystery_features Testing: {len(mysteryFeaturesTesting)}")

    # Saving
    np.save("mysteryFeaturesTraining.npy", mysteryFeaturesTraining)
    np.save("mysteryFeaturesValidation.npy", mysteryFeaturesValidation)
    np.save("mysteryFeaturesTesting.npy", mysteryFeaturesTesting)

    np.save("histFeaturesTraining.npy", histFeaturesTraining)
    np.save("histFeaturesValidation.npy", histFeaturesValidation)
    np.save("histFeaturesTesting.npy", histFeaturesTesting)

    histK = training(
        histFeaturesTraining, trainingLabels, histFeaturesValidation, validationLabels
    )
    print(f"Best k value for hist_features: {histK}")

    mysteryK = training(
        mysteryFeaturesTraining,
        trainingLabels,
        mysteryFeaturesValidation,
        validationLabels,
    )
    print(f"Best k value for mystery_features: {mysteryK}")

    print(f"Shape of histFeaturesTesting: {np.shape(histFeaturesTesting)}")
    print(f"Shape of histFeaturesTraining: {np.shape(histFeaturesTraining)}")
    print(f"Shape of testingLabels: {np.shape(testingLabels)}")
    print(f"Shape of trainingLabels: {np.shape(trainingLabels)}")

    histAccuracy = testing(
        histK, histFeaturesTesting, testingLabels, histFeaturesTraining, trainingLabels
    )

    mysteryAccuracy = testing(
        mysteryK,
        mysteryFeaturesTesting,
        testingLabels,
        mysteryFeaturesTraining,
        trainingLabels,
    )
