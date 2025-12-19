# Project 3: Retinal Disease Classification using CNN 👁️

## 📝 Topic Description
This project aims to build a **Convolutional Neural Network (CNN)** to automatically classify retinal fundus images into 8 distinct eye disease categories. The system is designed to assist in medical diagnosis by identifying multiple conditions from a single retinal image.

## 📊 Dataset Overview
- **Total Images**: 9,868 retinal fundus images (JPG/PNG).
- **Labels**: 8 disease categories.
- **Task**: Multi-label classification (an eye can have more than one disease).
- **Challenge**: The dataset is **highly imbalanced**, requiring specific optimization techniques.

### Disease Classes:
| Key | Disease Name |
|---|---|
| **N** | Normal |
| **D** | Diabetes |
| **G** | Glaucoma |
| **C** | Cataract |
| **A** | Age-related Macular Degeneration |
| **H** | Hypertension |
| **M** | Myopia |
| **O** | Others |

---

## 🛠️ CNN Architecture & Optimization
To satisfy the project requirements, the following implementation was developed:

### 1. CNN Model Architecture
- **Backbone**: **EfficientNetB0** (State-of-the-art CNN architecture).
- **Transfer Learning**: Pre-trained weights from ImageNet are used and then fine-tuned on medical data.

### 2. Optimization Techniques
*   **Data Augmentation**: Real-time transformations (flips, rotations, zooms) to increase dataset variety.
*   **Ben Graham's Preprocessing**: A specialized medical image technique using Gaussian Blur to highlight blood vessels and lesions.
*   **Focal Loss**: A loss function designed to handle **class imbalance** by focusing on "hard" minority cases.
*   **Dynamic Thresholding**: Optimizing decision boundaries per class to maximize the **F1-Score**.

---

## ⚙️ Setup & Reproduction

### 1. Data Structure on Google Drive
Upload your folder and rename it to `Project 03`. The structure must be:
```text
/MyDrive/
└── Project 03/
    ├── images/             # All JPG/PNG images
    └── label_images.csv    # CSV with 'images' and 'target' columns
2. Requirements
code
Bash
pip install tensorflow opencv-python pandas numpy matplotlib scikit-learn
```
📈 Evaluation Metrics
The model is evaluated using metrics suitable for imbalanced medical classification:
AUC-ROC: Measures the overall ability of the CNN to distinguish between classes.
F1-Score: Harmonic mean of Precision and Recall (crucial for imbalanced data).
Precision & Recall: Analyzed per disease category for reliability.
