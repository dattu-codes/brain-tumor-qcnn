# prediction_gradio.py

import os
import torch
from PIL import Image
import gradio as gr
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pennylane as qml

# safe checkpoint load (creates cleaned file if needed)
CLEAN_CHK = "qcnn_model_clean.pth"
RAW_CHK = "qcnn_model.pth"

def load_checkpoint(preferred=CLEAN_CHK, raw=RAW_CHK):
    if os.path.exists(preferred):
        try:
            return torch.load(preferred, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(preferred, map_location="cpu")
    if not os.path.exists(raw):
        raise FileNotFoundError("Checkpoint not found.")
    try:
        ck = torch.load(raw, map_location="cpu", weights_only=False)
    except Exception:
        ck = torch.load(raw, map_location="cpu")
    torch.save({"model_state": ck["model_state"], "class_names": ck["class_names"]}, preferred)
    try:
        return torch.load(preferred, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(preferred, map_location="cpu")

checkpoint = load_checkpoint()
class_names = checkpoint["class_names"]

# quantum setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (4, n_qubits, 3)}
q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

# model (must match training)
class HybridQCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            self.backbone = models.efficientnet_b0(pretrained=True)
        except Exception:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except Exception:
                self.backbone = models.efficientnet_b0(pretrained=False)
        # replace first conv to accept 1 channel
        try:
            f = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(1, f.out_channels,
                                                     kernel_size=f.kernel_size,
                                                     stride=f.stride,
                                                     padding=f.padding,
                                                     bias=(f.bias is not None))
        except Exception:
            self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridQCNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# prediction function
def predict(img):
    if img is None:
        return {c: 0.0 for c in class_names}
    img_pil = Image.fromarray(img).convert("L")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.exp(output).squeeze(0).cpu().numpy()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# launch gradio
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", image_mode="L"),
    outputs=gr.Label(num_top_classes=len(class_names)),
    title="Brain Tumor Classifier (EfficientNet + Quantum Layer)",
    description="Upload grayscale MRI brain images for classification."
).launch()
