# Image Classification with Support Vector Machine (SVM)
    This image classification project utilizes a **Support Vector Machine (SVM)** model combined with **Principal Component Analysis (PCA)** for dimensionality reduction. The system loads images from each class, performs data preprocessing, converts the data into a feature matrix, and applies scaling and PCA-based dimensionality reduction before training and evaluating the model.

# Project Structure
main
  |- Men vs Women
  |       |- men
  |       |- women
  |- data_load.py
  |- evaluate.py
## Dowload Dataset form :https://www.kaggle.com/datasets/playlist/men-women-classification?resource=download

## ⚙️ Features & Pipeline Overview
# 1. Automatic Class Detection (data_load.py): 
    Dynamically detects class subdirectories within data/. Reads supported image formats (.jpg, .jpeg, .png, .bmp) while gracefully skipping damaged/corrupted files.

# 2. Image Preprocessing (preprocess.py):
    - Converts color images (BGR) to Grayscale.
    - Resizes images to a uniform resolution of 100 x 100 pixels.
    - Flattens images into 10,000-dimensional feature vectors.
    - Normalizes pixel values from range [0, 255] to [0.0, 1.0].

# 3. Stratified Dataset Splitting (split_data.py):
    - Splits data into 80% Training and 20% Testing sets.
    - Preserves class distribution ratio across splits using  Stratified Sampling.

# 4. Dimensionality Reduction & Training (svm_model.py):
    - Integrates StandardScaler and PCA (default: 150 components with whitening) inside a scikit-learn Pipeline.
    - Fits a Support Vector Classifier on the PCA-transformed features.

# 5. Model Evaluation (evaluate.py):
    - Computes Accuracy, detailed Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.
    - Generates and exports a styled confusion_matrix.png heatmap.

# 6. Random Visual Inference (test_svm.py):
    - Randomly samples test images, predicts class labels, and renders a visual grid comparison with color-coded True vs. Predicted titles.