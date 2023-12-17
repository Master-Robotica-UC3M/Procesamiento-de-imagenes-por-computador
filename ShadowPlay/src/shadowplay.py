import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchmetrics

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################################
############################ CREAR EL DATASET ##############################
############################################################################

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, class_names, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]  # Assuming labels are in the 2nd column
        label = self.class_names.index(label)   # Convert label from string to numeric value
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor

        if self.transform:
            image = self.transform(image)

        return image, label


#Classes
# class_names = ('crab-shadow', 'fox-shadow', 'elephant-shadow', 'bird-shadow', 'dog-shadow', 'swan-shadow')
class_names = ('fox-shadow', 'elephant-shadow', 'bird-shadow', 'dog-shadow', 'swan-shadow')

# Define transformations (if needed)
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

# Path to your CSV file and image folder
csv_path = '../data/dataset/annotations.csv'
image_folder = '../data/dataset'  # Updated image folder path

# Create an instance of your custom dataset
custom_dataset = CustomDataset(csv_file=csv_path, root_dir=image_folder, class_names=class_names, transform=transform)

# Define the sizes for training and test sets
total_size = len(custom_dataset)
train_size = round(total_size*0.8)
test_size = total_size - train_size

# Split the dataset into training and test sets
train_set, test_set = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

batch_size = 1

# Create data loaders for training and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


############################################################################
############################ VISUALIZAR DATOS ##############################
############################################################################


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



# Obtener la etiqueta/clase correspondiente
train_class = train_label[0].item()  # Suponiendo que la etiqueta es un tensor de longitud 1
test_class = test_label[0].item()    # Suponiendo que la etiqueta es un tensor de longitud 1

# Mostrar la primera imagen del conjunto de entrenamiento y prueba con su título
print("Primera imagen del conjunto de entrenamiento:")
imshow(train_image[0], f"Clase: {class_names[train_class]}")

print("Primera imagen del conjunto de prueba:")
imshow(test_image[0], f"Clase: {class_names[test_class]}")



############################################################################
############################ DEFINIR LA RED ################################
############################################################################

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 101 * 101, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        # -> n, 3, 416, 416
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 206, 206
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 101, 101
        x = x.view(-1, 16 * 101 * 101)            # -> n, 163216
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 6
        return x
    

############################################################################
############################ ENTRENAR LA RED ###############################
############################################################################

# Hyper-parameters
num_epochs = 10
learning_rate = 0.01


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = criterion
num_outputs = 5
metrics_fn = torchmetrics.classification.MulticlassAccuracy(num_classes=num_outputs, average='micro').to(device)

def test(model, dataloader, loss_fn, metrics_fn):
   
    device = next(model.parameters()).device
    model.eval()
    metrics_fn = metrics_fn.to(device)
    metrics_fn.reset()
    with torch.no_grad():
        loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss += loss_fn(y_pred, y).item() * x.size(0)
            metrics_fn(y_pred, y)
        
        loss = loss / len(dataloader.dataset)
        metrics = metrics_fn.compute()

    return loss, metrics

n_total_steps = len(train_loader)

best_acc = 0
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

print("Todo listo jefe!")

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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

    

    # Evaluate the model after each epoch
    train_loss, train_metrics = test(model, train_loader, loss_fn, metrics_fn)
    test_loss, test_metrics = test(model, test_loader, loss_fn, metrics_fn)

    if best_acc <= train_metrics:
        best_acc = train_metrics
        torch.save(model.state_dict(), f'../models/best_model.pth')

    # Almacenamos los valores de pérdida y precisión en cada epoch
    train_loss_history.append(train_loss)
    train_acc_history.append(train_metrics)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_metrics)

    print(f'Epoch {epoch}, Precisión (train): {train_metrics:.4f}, Loss (train): {train_loss:.4f}')
    print(f'Epoch {epoch}, Precisión (test):  {test_metrics:.4f}, Loss (test):  {test_loss:.4f}')

print('Finished Training')

PATH = '../models/cnn.pth'
torch.save(model.state_dict(), PATH)


# Convertir listas a arrays de numpy para plotear
epochs = np.arange(1, num_epochs + 1)
test_loss_values = np.array(test_loss_history)
test_acc_values = np.array(test_acc_history)

# Gráfica para mostrar cómo evoluciona el loss con las epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, test_loss_values, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss vs Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Gráfica para mostrar cómo evoluciona la precisión con las epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, test_acc_values, label='Test Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs Epochs')
plt.legend()
plt.grid(True)
plt.show()


# Carga la imagen
image_path = 'test.jpeg'
image = Image.open(image_path)

# Transforma la imagen para que tenga el mismo formato que las imágenes con las que entrenaste tu modelo
transform = transforms.Compose([
    transforms.Resize((416, 416)),  # Asegúrate de redimensionarla a las dimensiones que acepta tu modelo
    transforms.ToTensor(),
])

# Aplica las transformaciones a la imagen
image = transform(image).unsqueeze(0)  # Añade una dimensión adicional para el batch

# Pasa la imagen por el modelo
with torch.no_grad():
    outputs = model(image)


# Obtén las probabilidades usando softmax
probabilities = F.softmax(outputs, dim=1)

# Obtén la clase predicha y su probabilidad asociada
predicted_prob, predicted_class = torch.max(probabilities, 1)
predicted_class = predicted_class.item()
predicted_prob = predicted_prob.item()

print("Clase predicha:", class_names[predicted_class])

test_image = Image.open('test.jpeg')
test_image = test_image.resize((416,416))
test_image = transform(test_image)
imshow(test_image, f"Predicted class: {class_names[predicted_class]}. Accuracy: {predicted_prob:.3f}")


