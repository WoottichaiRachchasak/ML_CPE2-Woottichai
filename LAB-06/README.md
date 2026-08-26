# LAB-06:NN(Men vs Women)
## Download Dataset from :[Download Dataset](https://www.kaggle.com/datasets/playlist/men-women-classification?resource=download)
## Project Structure
```text
LAB-06/
├── data/                    
│   ├── men/                 
│   └── women/               
├── outputs/                 
├── data_loader.py           
├── preprocessing.py         
├── split_data.py            
├── nn_model.py              
├── evaluate.py              
├── main.py                  
├── test_nn.py               
└── README.md    
 ``` 
## Pipeline Features   
### 1.Data Preparation & Preprocessing  (data_loader.py, preprocessing.py)
- Reads image files and resizes them to 100x100 pixels in RGB color space.
- Converts image collections into NumPy arrays 
- Performs 0-255 to 0-1 feature scaling using an internal Rescaling layer inside the model.  
### 2.Dataset Splitting (split_data.py)
- Splits data using a stratified ratio:
    - Train set: 70%
    - Validation set: 10%
    - Test set: 20%
### Model Architecture (nn_model.py) 
- Type: Fully-Connected Neural Network (MLP) 
- Input Layer: Rescaling(1/255) followed by Flatten() to reshape 2D images into 1D vectors 
- Hidden Layers:
    Dense Layer (256 units) + BatchNormalization + Dropout(0.4)  Dense Layer (128 units) + BatchNormalization + Dropout(0.4)  Dense Layer (64 units) + Dropout(0.3) 
- Output Layer: Dense Layer (Sigmoid activation for binary classification)  
- Optimizer: Adam (Learning Rate = 1e-4)  
- Loss Function: Binary Crossentropy  
- Callbacks: EarlyStopping and ReduceLROnPlateau to manage learning decay and prevent overfitting
## Usage
- Install library before run code 
From this command : pip install tensorflow opencv-python numpy scikit-learn matplotlib
- Run (main.py) for train and Evaluate model
- Run (test_nn.py) for Inference Test
Randomly selects 4 test samples, runs inference, prints prediction details, and saves the visualization grid to outputs/prediction_sample.png.
## Output Artifacts (outputs/)
Running the scripts generates the following artifacts:
- nn_model.keras: Saved trained Keras model.  
- history.json: Training and validation metrics history per epoch.  
- confusion_matrix.png: Confusion Matrix plot.  
- training_history.png: Accuracy and Loss learning curves.  
- prediction_sample.png: Sample grid of 4 predictions with confidence scores.  