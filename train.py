import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm  # Importar tqdm para la barra de progreso

# Definir un modelo CNN básico
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x

# Verificar si hay una GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# Ruta al archivo CSV de etiquetas y al directorio de imágenes
csv_file_path = 'data/Data_Entry_2017_v2020.csv'
image_dir = 'C:/Hackaton/data/images'

# Cargar el archivo CSV de etiquetas
df = pd.read_csv(csv_file_path)

# Procesar etiquetas
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
all_labels = np.unique([item for sublist in df['Finding Labels'] for item in sublist])

# Convertir etiquetas en formato binario para multi-etiqueta
for label in all_labels:
    df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)

# Seleccionar solo las imágenes con las etiquetas relevantes
df = df[df['Image Index'].apply(lambda x: os.path.exists(os.path.join(image_dir, x)))]

# Dividir el conjunto de datos en entrenamiento, validación y prueba
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

# Definir las transformaciones para las imágenes
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Crear una clase Dataset personalizada para PyTorch
class ChestXRayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.labels = df[all_labels].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.FloatTensor(self.labels[idx])
        return image, labels

# Crear los conjuntos de datos y cargadores de datos
train_dataset = ChestXRayDataset(train_df, image_dir, transform=image_transforms['train'])
val_dataset = ChestXRayDataset(val_df, image_dir, transform=image_transforms['val'])
test_dataset = ChestXRayDataset(test_df, image_dir, transform=image_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inicializar el modelo
model = SimpleCNN(num_classes=len(all_labels))
model = model.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.BCELoss()  # Para problemas de clasificación multietiqueta
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Función para entrenar el modelo
def train_model(model, criterion, optimizer, num_epochs=10):
    best_auc = 0.0
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Entrenamiento
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Entrenamiento Epoch {epoch+1}')):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Hacer predicciones
            outputs = model(inputs)

            # Calcular la pérdida
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Loss en el conjunto de entrenamiento: {running_loss/len(train_loader)}')

        # Validación
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Validación Epoch {epoch+1}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_labels.append(labels.cpu().numpy())
                val_preds.append(outputs.cpu().numpy())

        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)

        # Calcular AUC-ROC
        val_auc = roc_auc_score(val_labels, val_preds, average='macro')
        print(f'AUC en el conjunto de validación: {val_auc}')

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'model/best_model.pth')

        val_acc_history.append(val_auc)

    return model, train_acc_history, val_acc_history

# Entrenar el modelo
model, train_acc_history, val_acc_history = train_model(model, criterion, optimizer, num_epochs=10)

# Evaluar el modelo en el conjunto de prueba
model.load_state_dict(torch.load('model/best_model.pth'))
model.eval()

test_labels = []
test_preds = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Evaluación'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_labels.append(labels.cpu().numpy())
        test_preds.append(outputs.cpu().numpy())

test_labels = np.concatenate(test_labels)
test_preds = np.concatenate(test_preds)

test_auc = roc_auc_score(test_labels, test_preds, average='macro')
print(f'AUC en el conjunto de prueba: {test_auc}')

# Visualización de las curvas de entrenamiento
plt.plot(val_acc_history, label='Validación AUC')
plt.xlabel('Época')
plt.ylabel('AUC-ROC')
plt.title('Curva de AUC-ROC durante el entrenamiento')
plt.legend()
plt.show()
