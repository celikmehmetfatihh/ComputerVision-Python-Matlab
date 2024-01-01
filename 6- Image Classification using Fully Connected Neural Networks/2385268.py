import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, transforms
from PIL import Image
import matplotlib.pyplot as plt


"""DISCUSSION

Adding an extra layer improves the neural network's ability to capture complex patterns but increases computational requirements and the risk of overfitting. Consider task complexity and available resources when deciding between two or three layers.
The number of neurons in the hidden layers affects the network's capacity and ability to learn complex representations. More neurons can enhance performance, but excessive neurons may lead to overfitting. Finding the right balance is crucial for optimal results based on the task and available data.
"""

# Set random seed for reproducibility
torch.manual_seed(0)

# Set the device to either cuda if available, or Cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model using the nn.Module base class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the layers and operations of the model
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.softmax = nn.Softmax(dim=1)

    #  forward pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

# Custom Dataset class inherits from torch.utils.data.Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        # Initialize the dataset with features, labels, and an optional transform
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
         # Retrieve a specific item from the dataset based on the given index
        feature = self.features[index]
        label = self.labels[index]
        if self.transform:
            feature = self.transform(feature)
        return feature, label

# Function to load images from a directory
def imageLoader(path):
    images = []
    labels = []

    # Load all images with .jpg extension
    for file in os.scandir(path):
        if file.is_file() and file.name.endswith(".jpg"):
            imagePath = file.path
            image = Image.open(imagePath)
            images.append(image)

            # Extract class label from file name excluding numbers
            label = file.name.split(".")[0].rstrip("0123456789")
            labels.append(label)

    return images, labels

# Function to extract features from images
def featureExtractor(images, size):
    features = []

    for image in images:
        resizedImage = image.resize(size)
        resizedImage = resizedImage.convert("RGB")
        tensorImage = ToTensor()(resizedImage)
        features.append(tensorImage)

    return features

# Function to split the data into training, validation, and testing sets
def setFormer(features, labels):
    # Get the unique classes in the labels
    classes = sorted(set(labels))

    trainingFeatures = []
    trainingLabels = []
    validationFeatures = []
    validationLabels = []
    testingFeatures = []
    testingLabels = []

    for c in classes:
        # Get the features and labels for the current class
        classFeatures = [f for f, l in zip(features, labels) if l == c]
        classLabels = [l for l in labels if l == c]

        # Determine the number of samples for training, validation, and testing
        n = len(classFeatures)
        trainingNum = int(n * 0.5)
        validationNum = int(n * 0.25)
        trainingFeatures.extend(classFeatures[:trainingNum])
        trainingLabels.extend(classLabels[:trainingNum])
        validationFeatures.extend(classFeatures[trainingNum:trainingNum + validationNum])
        validationLabels.extend(classLabels[trainingNum:trainingNum + validationNum])
        testingFeatures.extend(classFeatures[trainingNum + validationNum:])
        testingLabels.extend(classLabels[trainingNum + validationNum:])

    # Define transformations for image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom datasets
    train_dataset = CustomDataset(trainingFeatures, trainingLabels, transform=transform)
    valid_dataset = CustomDataset(validationFeatures, validationLabels, transform=transform)
    test_dataset = CustomDataset(testingFeatures, testingLabels, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Initialize the CNN model
    model = CNN().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    train_loss_history = []
    valid_loss_history = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            # Update the total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")