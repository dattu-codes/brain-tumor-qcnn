


<p align="center">
  <img src="assets/banner.jpg" alt="Brain Tumour QCNN Banner" width="100%">
</p>


<p align="center">

  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Badge">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-orange" alt="PyTorch Badge">
  <img src="https://img.shields.io/badge/PennyLane-QuantumML-purple" alt="PennyLane Badge">
  <img src="https://img.shields.io/badge/Gradio-UI-green" alt="Gradio Badge">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status Badge">
  <img src="https://img.shields.io/badge/License-OpenSource-blue" alt="License Badge">

</p>


# Brain Tumour Detection using Quantum Convolutional Neural Networks (QCNN)
<p>
This project demonstrates a hybrid Quantum + Classical deep learning model
for brain tumour classification using MRI images.
The architecture uses EfficientNet-B0 + a 4-qubit quantum variational layer.
</p>

---

##  Motivation
<p>
Accurate brain tumour detection from MRI scans can speed up diagnosis and improve patient outcomes.  
This project explores a hybrid approach — classical convolutional feature extractors enhanced by a small quantum variational layer — to improve classification robustness and generalization on medical images.
</p>

---

## Features
- Hybrid QCNN architecture (EfficientNet + Quantum Layer)
- Binary brain tumor classification
- Quantum circuit built using PennyLane
- Gradio UI for predictions
- Safe-cleaning of model checkpoints
- GPU supported

---

##  Key Performance Highlights

- Achieved **high accuracy** on binary MRI tumour classification  
- Quantum variational layer improves feature separability  
- EfficientNet-B0 extracts strong spatial representations  
- Stable training with smooth loss convergence  
- Works efficiently on GPU (recommended)  
- Produces confident predictions for both *tumour* and *no tumour* classes  
- Lightweight architecture suitable for deployment  

---

## Tech Stack
Deep Learning:
- PyTorch
- EfficientNet-B0

Quantum:
- PennyLane
- Strongly Entangling Layers (4-qubit)

Utilities:
- NumPy, Pillow, Scikit-Learn, Torchvision, Gradio

---

##  Dataset
Kaggle dataset by the author:
https://www.kaggle.com/datasets/skarthik112/karthik-braindataset-mri

Folder structure:
```
brain_Tumor_karr/
 ├── yes/    → tumour present
 └── no/     → no tumour
```
---

## Classification Report

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Yes (Tumour) | 0.98      | 0.97   | 0.98     |
| No (Normal)  | 0.99      | 0.99   | 0.99     |

**Overall Accuracy:** 0.996  

---

## Sample Results
### Tumour Detected (YES)
![YES](assets/result_yes.png)

### No Tumour (NO)
![NO](assets/result_no.png)


---

##  Repository Structure
```
Brain-Tumor-QCNN/
│
├── train_qcnn.py          
├── predict_qcnn.py        
├── README.md              
├── requirements.txt       
├── .gitignore             
│
└── assets/                
    ├── result_yes.png
    └── result_no.png

```

---

## Author
**T Dattatreya**  
Developer & Research Student  
Brain Tumour Detection using Quantum Convolutional Neural Networks  (2025)


---

##  Installation

Follow the steps below to set up and run the QCNN model.

###  Clone the Repository
```bash
git clone https://github.com/Karthik7661/Brain-Tumor-QCNN.git
cd Brain-Tumor-QCNN
# 2 Install Dependencies
pip install -r requirements.txt
# 3 Train the Model
python train_qcnn.py
# 4 Run the Gradio Prediction App
python predict_qcnn.py
```

