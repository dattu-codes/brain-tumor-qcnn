import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import pennylane as qml
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 1. Dataset Path - update this with your dataset folder path
dataset_path = "data/brain_dataset"  # Expected subfolders: 'no_tumor', 'glioma', 'meningioma', 'pituitary'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# 2. Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
full_dataset = ImageFolder(root=dataset_path, transform=transform)
class_names = full_dataset.classes
print("Classes found:", class_names)

# Split data
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 3. Quantum Circuit Setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (4, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# 4. Hybrid Model: EfficientNet Backbone + Quantum Layer
class HybridQCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust for grayscale
        self.backbone.classifier = nn.Identity()
        self.fc1 = nn.Linear(1280, n_qubits)
        self.q_layer = q_layer
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.tanh(self.fc1(x))
        x = self.q_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = HybridQCNN(num_classes=len(class_names))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# Optimizer, loss and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.NLLLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
epochs = 50

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(imgs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            loss = loss_fn(output, labels)
            val_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            val_preds.extend(preds.cpu())
            val_labels.extend(labels.cpu())
    avg_val_loss = val_loss / len(test_loader)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch + 1}/{epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}")

print("Classification Report:")
print(classification_report(val_labels, val_preds, target_names=class_names))

# Save model
torch.save({
    "model_state": model.state_dict(),
    "class_names": class_names,
    "transform": transform
}, "qcnn_model.pth")
print("Model saved as qcnn_model.pth")
