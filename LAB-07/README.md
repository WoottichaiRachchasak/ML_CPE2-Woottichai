# LAB-06:NN(Men vs Women)
## Download Dataset from :[Download Dataset](https://www.kaggle.com/datasets/playlist/men-women-classification?resource=download)
```text
LAB-07/
├── data/                    
│   ├── men/                 
│   └── women/               
├── outputs/                 
├── cnn_model.py           
├── data_loader.py         
├── evaluate.py            
├── main.py              
├── preprocessing.py              
├── README.md                  
├── split_data.py               
└── test_cnn.py    
 ``` 
 
 ## Pipeline Features
### 1. Data Preparation & Preprocessing (data_loader.py, preprocessing.py)
- Reads image files and resizes them to 150x150 pixels in RGB color space.
- Converts image collections into NumPy arrays.
- Performs 0-255 to 0-1 feature scaling using an internal Rescaling layer inside the model.
### 2. Dataset Splitting (split_data.py)
- Splits data using a stratified ratio:
    - Train set: 70%
    - Validation set: 10%
    - Test set: 20%
### 3. Model Architecture (cnn_model.py)
- Type: Convolutional Neural Network (CNN)
- Input Layer: Rescaling(1/255) followed by Data Augmentation (RandomFlip, RandomRotation, RandomZoom).
- Hidden Layers:
    - Conv2D (32 filters) + BatchNormalization + MaxPooling2D
    - Conv2D (64 filters) + BatchNormalization + MaxPooling2D
    - Conv2D (128 filters) + BatchNormalization + MaxPooling2D
    - GlobalAveragePooling2D + Dropout(0.3)
    - Dense Layer (128 units, ReLU) + Dropout(0.3)
- Output Layer: Dense Layer (Sigmoid activation for binary classification).
### 4. Training & Callbacks (cnn_model.py)
- Optimizer: Adam (Learning rate = 1e-4)
- Loss Function: Binary Crossentropy
- Callbacks:
    - EarlyStopping (stops when validation loss stops improving for 5 epochs, restoring best weights)
    - ReduceLROnPlateau (halves the learning rate if validation loss plateaus for 3 epochs, min_lr = 1e-5)
### 5. Evaluation & Testing (evaluate.py, test_cnn.py)
- Calculates accuracy, classification report, and confusion matrix.
- Plots training accuracy and loss curves, and generates sample prediction grid images.