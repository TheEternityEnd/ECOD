import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Definir la clase del modelo SimpleCNN que usaste durante el entrenamiento
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
        x = x.view(-1, 128 * 28 * 28)  # Aplanar el tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid para clasificación multietiqueta
        return x

# Inicializar el modelo y cargar los pesos guardados
num_classes = 15  # Ajustar este valor al número de etiquetas
model = SimpleCNN(num_classes)

# Cargar los pesos del modelo guardados en best_model.pth
model.load_state_dict(torch.load('model/best_model.pth', map_location=torch.device('cpu')))
model.eval()  # Poner el modelo en modo de evaluación

# Definir las transformaciones para preprocesar las imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Mismos valores usados en el entrenamiento
])

# Función para cargar y preprocesar una imagen
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Añadir una dimensión para el batch

# Ruta a la nueva imagen de radiografía que deseas predecir
image_path = 'C:/Hackaton/data/images/00000001_000.png'  # Cambia a la ruta de tu imagen
image = preprocess_image(image_path)

# Hacer la predicción en la nueva imagen
with torch.no_grad():
    output = model(image)

# Lista de etiquetas de "Finding Labels" en el mismo orden en que las entrenaste
labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
          'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
          'Pleural_Thickening', 'Hernia']

# Convertir las predicciones a un formato legible
predictions = torch.sigmoid(output).squeeze().cpu().numpy()

# Mostrar todas las etiquetas con sus probabilidades
print("Todas las predicciones con sus probabilidades:")
for label, pred in zip(labels, predictions):
    print(f"{label}: {pred:.4f}")

# Seleccionar la etiqueta con la mayor probabilidad
predicted_label_index = torch.argmax(output, dim=1).item()
predicted_label = labels[predicted_label_index]

# Mostrar la etiqueta con la probabilidad más alta
print(f"\nEtiqueta predicha con mayor probabilidad: {predicted_label} ({predictions[predicted_label_index]:.4f})")
