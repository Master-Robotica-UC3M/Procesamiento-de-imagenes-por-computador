import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

# Define la estructura de tu modelo. Por ejemplo:
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

# Instancia el modelo
model = ConvNet()

# Ruta donde está guardado el modelo
PATH = './cnn.pth'

# Carga los parámetros guardados en el modelo
model.load_state_dict(torch.load(PATH))
# Asegúrate de poner model.eval() si estás en modo de evaluación (no entrenamiento)
# model.eval()

# Ahora el modelo 'model' contiene los parámetros que guardaste previamente y está listo para ser utilizado.

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

# Obtén la clase predicha
class_names = ('crab', 'fox', 'elephant', 'bird', 'dog', 'swan')

# Obtén las probabilidades usando softmax
probabilities = F.softmax(outputs, dim=1)

# Obtén la clase predicha y su probabilidad asociada
predicted_prob, predicted_class = torch.max(probabilities, 1)
predicted_class = predicted_class.item()
predicted_prob = predicted_prob.item()

print("Clase predicha:", class_names[predicted_class])

test_image = cv2.imread('test.jpeg')
test_image = cv2.resize(test_image, (416,416), interpolation=cv2.INTER_LINEAR)
print(predicted_class)
cv2.imshow(f"Predicted class: {class_names[predicted_class]}. Accuracy: {predicted_prob:.3f}",test_image)
cv2.waitKey(0)
