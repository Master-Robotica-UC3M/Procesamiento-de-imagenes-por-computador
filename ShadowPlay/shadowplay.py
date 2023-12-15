import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = self.create_label_map()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 3]  # Assuming labels are in the 4th column
        label = self.label_map[label]   # Convert label from string to numeric value
        label_tensor = torch.tensor(label, dtype=torch.long)  # Convert label to tensor

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

    def create_label_map(self):
        unique_labels = self.data.iloc[:, 3].unique()  # Get unique labels from the column
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        return label_map

#Classes
class_names = ('crab', 'fox', 'elephant', 'bird', 'dog', 'swan')

# Define transformations (if needed)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    # Add more transformations as needed (e.g., normalization)
])

# Path to your CSV file and image folder
csv_path = '/content/gdrive/MyDrive/Colab Notebooks/data/data/_annotations.csv'
image_folder = '/content/gdrive/MyDrive/Colab Notebooks/data/data/'  # Updated image folder path

# Create an instance of your custom dataset
custom_dataset = CustomDataset(csv_file=csv_path, root_dir=image_folder, transform=transform)

# Define the sizes for training and test sets
train_size = 40
test_size = 10

# Split the dataset into training and test sets
train_set, test_set = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

batch_size = 1

# Create data loaders for training and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# Obtener la primera imagen y etiqueta del conjunto de entrenamiento
train_iter = iter(train_loader)
train_image, train_label = next(train_iter)

# Obtener la primera imagen y etiqueta del conjunto de prueba
test_iter = iter(test_loader)
test_image, test_label = next(test_iter)

# Función para mostrar la imagen y su etiqueta
def imshow(image, title):
    plt.imshow(image.permute(1, 2, 0))  # Cambiar dimensiones para visualización
    plt.title(title)
    plt.axis('off')
    plt.show()

print(train_label)

# Obtener la etiqueta/clase correspondiente
train_class = train_label[0].item()  # Suponiendo que la etiqueta es un tensor de longitud 1
test_class = test_label[0].item()    # Suponiendo que la etiqueta es un tensor de longitud 1

# Mostrar la primera imagen del conjunto de entrenamiento y prueba con su título
print("Primera imagen del conjunto de entrenamiento:")
imshow(train_image[0], f"Clase: {train_class}")

print("Primera imagen del conjunto de prueba:")
imshow(test_image[0], f"Clase: {test_class}")


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 101 * 101, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        # -> n, 3, 416, 416
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 206, 206
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 101, 101
        x = x.view(-1, 16 * 101 * 101)            # -> n, 163216
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 6
        return x
    

# Hyper-parameters
num_epochs = 50
batch_size = 1
learning_rate = 0.01


# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(6)]
    n_class_samples = [0 for i in range(6)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    print(n_class_samples)
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(6):
      if n_class_samples[i] > 0:
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {class_names[i]}: {acc} %')