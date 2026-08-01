import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset_dir = "UTKFace"
images = []
ages = []

file_name = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
random.seed(42)
random.shuffle(file_name)
file_name = file_name[:1000]

print("Reading and converting image data...")
for file_name in file_name:
    p = file_name.split('_')
    if len(p) >= 4:
        age = int(p[0])

        img_path = os.path.join(dataset_dir,file_name)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img,(64,64))
            images.append(img.flatten())
            ages.append(age)

X = np.array(images) / 255.0
y = np.array(ages)

print(f"read data successfully!!")
print(f"- total images: {X.shape[0]} ภาพ")
print(f"- Number of original features before PCA: {X.shape[1]} px\n")

n_components = 20 
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)

print(f"- Number of features after dimensionality reduction with PCA: {X_pca.shape[1]} Components\n")


X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

X_train_simple = X_train[:, :1] 
X_test_simple = X_test[:, :1]

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)


y_pred_simple = simple_model.predict(X_test_simple)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)


y_pred_multi = multi_model.predict(X_test)


rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))
rmse_multi = np.sqrt(mean_squared_error(y_test, y_pred_multi))


r2_simple = r2_score(y_test, y_pred_simple)
r2_multi = r2_score(y_test, y_pred_multi)

print("="*60)
print("             Performance Comparison Results (LAB 1)")
print("="*60)
print(f"1. Simple Linear Regression   (1  PC)  -> RMSE: {rmse_simple:.2f} years | R2 Score: {r2_simple:.4f}")
print(f"2. Multiple Linear Regression (20 PCs) -> RMSE: {rmse_multi:.2f} years | R2 Score: {r2_multi:.4f}")
print("="*60)


plt.figure(figsize=(12, 5))

# Simple Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_simple, alpha=0.4, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
plt.title(f'Simple Linear Regression\nRMSE: {rmse_simple:.2f} | R2: {r2_simple:.4f}')
plt.xlabel('Actual Age ')
plt.ylabel('Predicted Age')
plt.grid(True)

# Multiple Linear Regression
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_multi, alpha=0.4, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
plt.title(f'Multiple Linear Regression\nRMSE: {rmse_multi:.2f} | R2: {r2_multi:.4f}')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.grid(True)

plt.tight_layout()
plt.show()