# Machine Learning Lab_04: Classification Model (Wine Quality Dataset)

## Directory Structure & File Overview

### `data_loader.py`
- Loads the dataset from `WineQT.csv`.
-  Sets the `quality` column as the target variable and encodes it into discrete class labels.
-  Extracts the 11 chemical features into a feature matrix.
-  Splits data into Train (60%), Validation (20%), and Test (20%) sets using stratified sampling to preserve class distribution.
-  Performs feature scaling via `StandardScaler` fitted on the  training set and transforms validation and test sets accordingly.

### `knn_tf.py`
 - Implements the `TFKNNClassifier` class using TensorFlow  operations.
 - Calculates Euclidean distances between test samples and training samples.
 - Selects the $k$-nearest neighbors using `tf.math.top_k`.
 - Determines class predictions via majority voting with one-hot encoding.

### **`evaluate.py`**
-  Provides evaluation and visualization utilities, saving outputs to the `outputs/` directory.
-  `plot_k_curve()`: Plots validation accuracy across different $k$ values (`01_k_curve.png`).
-  `plot_confusion_matrix()`: Generates a confusion matrix comparing actual vs. predicted labels (`02_confusion_matrix.png`).
-  `print_report()`: Prints detailed class-wise metrics (Precision, Recall, F1-score).
-  `save_predictions()`: Exports individual prediction results to a CSV file (`predictions.csv`).

### **`main.py`**
  - Main execution script covering 5 workflow steps:
    1. **Load Data**: Invokes `data_loader` to prepare and scale data.
    2. **Hyperparameter Tuning**: Iterates through candidate $k$ values on the validation set to find $best\_k$.
    3. **Model Evaluation**: Tests the model using $best\_k$ on the test set and outputs classification reports.
    4. **Algorithm Verification**: Compares TensorFlow model predictions against Scikit-learn's `KNeighborsClassifier`.
    5. **Baseline Comparison**: Benchmarks model performance against a majority-class baseline.