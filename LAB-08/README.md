# VGG-Based Deep Convolutional Neural Network (DCNN) for Image Classification

## Download Dataset from :[Download Dataset](https://www.kaggle.com/datasets/playlist/men-women-classification?resource=download)

## Project structure


```text
├── data/                  # Directory containing dataset subfolders organized by class (men/women)
├── outputs/               # Directory storing trained models, numpy arrays, and evaluation plots
├── data_loader.py         # Automated dataset loader with image filtering and validation
├── preprocessing.py       # Color space conversion (BGR to RGB) and image resizing
├── split_data.py          # Stratified dataset splitting (Train / Validation / Test)
├── vgg_model.py           # VGG architecture definition, callbacks, and training loop
├── evaluate.py            # Model performance evaluation (Accuracy, F1-Score, Confusion Matrix)
├── test_vgg.py            # Script for testing and visualizing sample predictions
└── main.py                # Main orchestration script running the full end-to-end pipeline

```

## Pipeline Features & Architecture
# 1. Data Preparation & Preprocessing (data_loader.py, preprocessing.py)
- Reads image files from directories, converts color space to RGB,  and resizes images to a standard 150x150 pixels.
- Automatically filters out and skips corrupted or unreadable image files.
- Converts the entire dataset into structured NumPy arrays.

# 2. Dataset Splitting (split_data.py)
- Splits data using a stratified ratio to preserve class balance:
    - Train Set: 70% (used for model training)

    - Validation Set: 10% (used for monitoring during training)

    - Test Set: 20% (used for final evaluation)

# 3. Model Architecture (vgg_model.py)
- Architecture: VGG-Based Deep CNN (VGG_SMALL / VGG16 Layout)
- Input Layer: Includes built-in Rescaling(1.0 / 255) and Data Augmentation layers (RandomFlip, RandomRotation, RandomZoom) active exclusively during training.
- Hidden Layers: Convolutional blocks stacked with BatchNormalization and MaxPooling, followed by GlobalAveragePooling2D, a Dense Layer (128 units, ReLU), and Dropout (0.5) for overfitting prevention.
- Output Layer: Dense Layer (Sigmoid Activation for binary classification).

# 4. Training & Callbacks (vgg_model.py)
- Optimizer: Adam (Learning Rate = 1e-4)
- Loss Function: Binary Crossentropy
- Training Control Callbacks:
    - EarlyStopping: Automatically halts training when val_loss stops improving for a specified patience and restores the best model weights (restore_best_weights=True).
    - ReduceLROnPlateau: Halves the learning rate (factor=0.5) when validation loss plateaus.

# 5. Evaluation & Testing (evaluate.py, test_vgg.py)
- Calculates accuracy, generates a Classification Report (Precision, Recall, F1-score), and computes the Confusion Matrix.
- Generates training history curves (training_history.png) and random sample prediction grid images with confidence percentages (prediction_sample.png).

##  System Requirements
Ensure Python and the following libraries are installed:
* `tensorflow` (>= 2.x)
* `scikit-learn`
* `opencv-python`
* `matplotlib`
* `numpy`

## How to Run
### Execute the main script to run the complete pipeline (data loading, preprocessing, training, and evaluation):
- python main.py
### Test the trained model against random samples from the test set:
- python test_vgg.py

## (Outputs)
Upon successful execution, generated outputs are automatically saved in the `outputs/` directory:

* **`vgg_model.keras`**: Trained model weights and architecture.

* **`confusion_matrix.png`**: Confusion matrix visualization of prediction results.

* **`training_history.png`**: Comparative line plots of Accuracy and Loss between Train and Validation sets across epochs.

* **`prediction_sample.png`**: Sample prediction images complete with model confidence percentages.