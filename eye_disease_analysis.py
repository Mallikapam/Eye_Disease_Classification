#import numpy as np 
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models 
import torchvision.transforms
from torchvision.transforms import ToTensor
import os
from PIL import Image 
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm


def open_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# define dataset directory 
path = 'retina_dataset/dataset'
#img_size = 224 
#batch_size = 32

image = Image.open('retina_dataset/dataset/1_normal/NL_001.png')
#image.show()

INPUT_HEIGHT = 204
INPUT_WIDTH = 308

#             

image_transforms = transforms.Compose([
    transforms.Resize((204, 308)),  #1232, 816
    transforms.CenterCrop(size=(204, 206)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4563, 0.2717, 0.1612], std=[0.5519, 0.3326, 0.2021])
    #transforms.Normalize(mean=[0.3066, 0.1828, 0.1091], std=[0.324, 0.1947, 0.1162])
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = datasets.ImageFolder(root=path, transform=image_transforms)

#tensor_to_image(dataset[0][0][0])
#tensor_to_image(dataset[0][0][1])
#tensor_to_image(dataset[0][0][2])

#  torch.mean(a)

# print(dataset[0][0])
# transform = torchvision.transforms.ToPILImage()   # <----------------------------- to see images 
# img = transform(dataset[0][0])
# img.show()

# num_pixels = 0
# sum_rgb = torch.zeros(3)
# sum_squared_rgb = torch.zeros(3)

# for img, _ in dataset:
#     num_pixels += img.shape[1] * img.shape[2]
#     sum_rgb += img.sum(dim=[1,2])
#     sum_squared_rgb += (img ** 2).sum(dim=[1,2])

# mean_rgb = 0
# std_rgb = 0
# mean_rgb = sum_rgb/(204 * 206 * 601)
# std_rgb = (sum_squared_rgb / ((204 * 206 * 601) - mean_rgb ** 2)).sqrt()
# print(mean_rgb)
# print(std_rgb)


# blueMean = 0
# greenMean = 0
# redMean = 0

# bluestd = 0
# greenstd = 0
# redstd = 0
# print(len(dataset))
# for i in dataset:
#     redMean += torch.mean(i[0][0])
#     greenMean += torch.mean(i[0][1])
#     blueMean += torch.mean(i[0][2])
#     redstd += torch.std(i[0][0])
#     greenstd += torch.std(i[0][1])
#     bluestd += torch.std(i[0][2])
    

# print("red mean:", redMean)
# print("green mean:", greenMean)
# print("blue mean:", blueMean)
# print("red std:", redstd)
# print("green std:", greenstd)
# print("blue std:", bluestd)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# transform = torchvision.transforms.ToPILImage()
# img = transform(train_loader.dataset[0][0])
# img.show()
# print(train_loader.dataset[0][1])


# Input Image -> Conv2d -> [Activation] -> [Pooling] -> Conv2d -> 
# [Activation] -> [Pooling] -> ... -> Flatten -> Fully Connected Layer ->
# [Activation] -> Fully Connected Layer -> Output (Class Probabilities)

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):  # num class = 10
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        in_channels: 3 
        num_classes: 4
        super(CNN, self).__init__()

        # First convolutional layer: 3 input channel, 8 output channels, 10x10 kernel, stride 1, padding 0
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=15, stride=1, padding=0)
        # next convolutional layers in_channels must be equal to the out_channels of the immediately preceding layer

        self.bn1 = nn.BatchNorm2d(8) 
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 8x8 kernel, stride 1, padding 3
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=10, stride=1, padding=3)
        # Fully connected layer: 32*13*13 input features 

        self.bn2 = nn.BatchNorm2d(16)
      
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=2, padding=5)

        self.bn3 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(p=0.5)

        dummy_input = torch.randn(1, in_channels, 204, 206)

        # Pass the dummy tensor through the convolutional and pooling layers to get the size before flattening
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(dummy_input))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)))) 


        num_features_before_fc = x.numel() // x.shape[0] 

        print(f"Calculated in_features for fc1: {num_features_before_fc}")

        self.fc1 = nn.Linear(in_features=num_features_before_fc, out_features=num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """

        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)))) 
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x)))) 
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x)))) 

        x = x.reshape(x.shape[0], -1)  
        x = self.dropout(x)            
        x = self.fc1(x)                 

        # x = F.relu(self.conv1(x))  
        # x = self.pool(x)           
        # x = F.relu(self.conv2(x))  
        # x = self.pool(x)           
        # x = F.relu(self.conv3(x))
        # x = self.pool(x)
        # x = x.reshape(x.shape[0], -1)  
        # x = self.fc1(x)            
        return x

#print(train_loader.dataset[0].size())
input_size = 126072        #204*206*3
num_classes = 4  # 4 types of eye: normal, cataracts, glaucoma, retina disease
learning_rate = 0.001
batch_size = 32
num_epochs = 75  

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(in_channels=3, num_classes=num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#tensor_to_image(train_loader.dataset[0][0][1])


for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass: compute the model output
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()



def check_accuracy(loader, model, dataset_type):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    # if loader.dataset.train:
    #     print("Checking accuracy on training data")
    # else:
    #     print("Checking accuracy on test data")

    print(f"Checking accuracy on {dataset_type} data") 

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    
    model.train()  

# Final accuracy check on training and test sets
check_accuracy(train_loader, model, 'training')
check_accuracy(test_loader, model, 'test')