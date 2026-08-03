# Download the dataset used in this lab : https://www.kaggle.com/datasets/jangedoo/utkface-new

# Age Prediction using PCA and Linear Regression

This project demonstrates an end-to-end Machine Learning pipeline for **predicting age from facial images**. It utilizes **Principal Component Analysis (PCA)** for dimensionality reduction and evaluates performance by comparing a **Simple Linear Regression** baseline against a **Multiple Linear Regression** model.

---

## Objectives
* Preprocess facial images and perform flattening from 2D pixel arrays to 1D feature vectors.
* Reduce high-dimensional image feature space using **Principal Component Analysis (PCA)**.
* Train and evaluate **Simple Linear Regression** (1 PC) versus **Multiple Linear Regression** (20 PCs).
* Measure model accuracy using standard regression evaluation metrics.

---

## Tech Stack & Libraries
* **Language:** Python 3.x
* **Libraries:**
  * `numpy` – Multi-dimensional array handling and mathematical operations
  * `scikit-learn` – PCA transformation, data splitting (`train_test_split`), linear regression models, and metric evaluations
  * `opencv-python` / `PIL` – Image loading and resizing
  * `matplotlib` – Data visualization and plotting

---

## 🔄 Machine Learning Pipeline

The project workflow consists of five main steps:

```text
[Input Images] ──> 1. Flattening ──> 2. PCA (20 PCs) ──> 3. Train/Test Split ──> 4. Model Training ──> 5. Evaluation
```
## 1. Data Preprocessing & Flattening:
    - Load facial images and resize them to a uniform resolution.
    - Flatten 2D image arrays(64*64 px) into 1D vectors of length 4096.

## 2. Dimensionality Reduction (PCA):
    - Apply PCA to reduce features from $4,096$ pixels down to 20 Principal Components (PCs) to remove noise and compress feature representation.

## Data Splitting:
    - Partition the dataset into training (80๔) and testing (20%) sets using train_test_split (test_size=0.2, random_state=42).

## Model Training & Comparison:
    - Simple Model: Slices only the first component (X_train[:, :1]) to train a single-feature Linear Regression baseline.
    - Multiple Model: Uses all 20 principal components (X_train) to train a Multiple Linear Regression model.

## Model Evaluation:
    - Pass unseen test data (X_test) into .predict() and compare predictions against ground truth labels (y_test).

## Evaluation & Metrics
The models are evaluated using two primary metrics:
    - RMSE (Root Mean Squared Error): Represents the average prediction error in years (lower is better).
    - $R^2$ Score (Coefficient of Determination): Measures how well the model explains variance in the target variable ($0.0$ to $1.0$, closer to 1.0 is better).


## Conclusion
    Using Multiple Linear Regression (20 PCs) captures significantly more spatial structural detail from facial images compared to a single PC. This leads to a substantial reduction in RMSE and a higher $R^2$ Score, demonstrating the importance of retaining sufficient principal components for complex image regression tasks.