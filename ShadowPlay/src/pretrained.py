import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
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
        label = self.data.iloc[idx, 3]  # Assuming labels are in the 4th column
        label = self.class_names.index(label)   # Convert label from string to numeric value
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor

        if self.transform:
            image = self.transform(image)

        return image, label

class_names = ('crab-shadow', 'fox-shadow', 'elephant-shadow', 'bird-shadow', 'dog-shadow', 'swan-shadow')

# Define transformations (if needed)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of ResNet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization values for ImageNet
])

# Path to your CSV file and image folder
csv_path = '/home/gonecho/Documents/MasterRobotica/PIC_TrabajoFinal/ShadowPlay/data/_annotations.csv'
image_folder = '/home/gonecho/Documents/MasterRobotica/PIC_TrabajoFinal/ShadowPlay/data'  # Updated image folder path

# Create an instance of your custom dataset
custom_dataset = CustomDataset(csv_file=csv_path, root_dir=image_folder, class_names=class_names, transform=transform)

# Define the sizes for training and test sets
train_size = 40
test_size = 10

# Split the dataset into training and test sets
train_set, test_set = torch.utils.data.random_split(custom_dataset, [train_size, test_size])

batch_size = 4  # Adjust batch size based on available memory

# Create data loaders for training and test sets
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

############################################################################
############################ DEFINIR LA RED ################################
############################################################################

model = models.resnet18(pretrained=True)  # Load the pre-trained ResNet18 model
num_features = model.fc.in_features  # Get the number of input features for the last fully connected layer
model.fc = nn.Linear(num_features, len(class_names))  # Modify the last layer for 6 classes

# Congelar todas las capas excepto la capa final
for name, param in model.named_parameters():
    if "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model = model.to(device)

# Hyper-parameters
num_epochs = 50
learning_rate = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = criterion
num_outputs = len(class_names)
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

best_acc = 0
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

for epoch in range(num_epochs):
    model.train()
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

    # Evaluate the model after each epoch
    train_loss, train_metrics = test(model, train_loader, loss_fn, metrics_fn)
    test_loss, test_metrics = test(model, test_loader, loss_fn, metrics_fn)

    if best_acc <= train_metrics:
        best_acc = train_metrics
        torch.save(model.state_dict(), 'best_model.pth')

    # Store loss and accuracy values
    train_loss_history.append(train_loss)
    train_acc_history.append(train_metrics)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_metrics)

    print(f'Epoch {epoch}, Train Accuracy: {train_metrics:.4f}, Train Loss: {train_loss:.4f}')
    print(f'Epoch {epoch}, Test Accuracy: {test_metrics:.4f}, Test Loss: {test_loss:.4f}')

print('Finished Training')

# Plotting
epochs = np.arange(1, num_epochs + 1)
test_loss_values = np.array(test_loss_history)
test_acc_values = np.array(test_acc_history)

plt.figure(figsize=(10, 5))
plt.plot(epochs, test_loss_values, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss vs Epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, test_acc_values, label='Test Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy vs Epochs')
plt.legend()
plt.grid(True)
plt.show()


# Función para mostrar la imagen y su etiqueta
def imshow(image, title):
    plt.imshow(image.permute(1, 2, 0))  # Cambiar dimensiones para visualización
    plt.title(title)
    plt.axis('off')
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

