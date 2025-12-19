Project 3: Retinal Disease Classification using CNN
📝 Topic Description
This project aims to build a Convolutional Neural Network (CNN) to automatically classify retinal fundus images into 8 distinct eye disease categories. The system is designed to assist in medical diagnosis by identifying multiple conditions from a single retinal image.
📊 Dataset Overview
Total Images: 9,868 retinal fundus images (JPG/PNG).
Labels: 8 disease categories.
Task: Multi-label classification (an eye can have more than one disease).
Challenge: The dataset is highly imbalanced, requiring specific optimization techniques to ensure the model doesn't just bias towards the majority class.
Disease Classes:
N: Normal
D: Diabetes
G: Glaucoma
C: Cataract
A: Age-related Macular Degeneration
H: Hypertension
M: Myopia
O: Others
🛠️ CNN Architecture & Optimization
To satisfy the project requirements, the following implementation was developed:
1. CNN Model Architecture
Instead of a basic CNN, this project utilizes EfficientNetB0 as the backbone.
EfficientNet is a state-of-the-art CNN that scales depth, width, and resolution efficiently.
Transfer Learning: Pre-trained weights from ImageNet are used and then fine-tuned on the medical dataset.
2. Optimization Techniques (Handling Imbalance & Accuracy)
We applied four major optimization techniques:
Data Augmentation: Real-time transformations including random flips, rotations, zooms, and contrast adjustments to increase dataset variety and prevent overfitting.
Ben Graham's Preprocessing: A domain-specific medical image technique that uses Gaussian Blur to remove noise and highlight blood vessels/lesions.
Focal Loss: A specialized loss function designed for imbalanced datasets. It down-weights easy-to-classify examples and focuses the model's learning on "hard" minority cases.
Dynamic Thresholding: Instead of a fixed 0.5 threshold, we optimize the decision boundary for each class to maximize the F1-Score.
⚙️ Setup & Reproduction
1. Data Structure on Google Drive
Upload the dataset folder to your Google Drive and rename it to Project 03. The structure must be:
/MyDrive/
└── Project 03/
    ├── images/             # All JPG/PNG images
    └── label_images.csv    # CSV with 'images' and 'target' columns
2. Requirements
pip install tensorflow opencv-python pandas numpy matplotlib scikit-learn
3. Training Workflow
The training script executes in two stages:
Stage 1: Warm-up the classification head (Freezing the CNN backbone).
Stage 2: Fine-tuning (Unfreezing the top layers of the CNN for specialized medical feature extraction).
📈 Evaluation Metrics
Given the imbalanced nature of the dataset, the model is evaluated using:
AUC-ROC (Area Under Curve): Measures the overall ability of the CNN to distinguish between classes.
F1-Score: The harmonic mean of Precision and Recall (crucial for imbalanced medical data).
Precision & Recall: Analyzed per disease category to ensure high diagnostic reliability.
