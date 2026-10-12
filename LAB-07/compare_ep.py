import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from cnn_model import build_model
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR,  "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# โหลดข้อมูลเตรียมไว้ครั้งเดียว
print("Loading data for comparison...")
images, labels, classes = load_data(DATA_PATH, 150, 3000)
X = to_features(images)
y = labels
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, 0.2, 0.1)

# กำหนดจำนวน Epochs ที่ต้องการเทียบ
epoch_list = [10, 20, 30]
history_results = {}

for ep in epoch_list:
    print(f"\n--- Training with {ep} Epochs ---")
    model = build_model(X_train.shape[1:], len(classes))
    
    # รันเทรนตามจำนวน Epochs ที่กำหนด
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=ep,
        batch_size=32,
        verbose=1
    )
    history_results[ep] = history.history

# สร้างกราฟเส้นเปรียบเทียบรวมกัน
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, ep in enumerate(epoch_list):
    h = history_results[ep]
    # พล็อต Accuracy
    axes[0].plot(range(1, len(h['accuracy']) + 1), h['accuracy'], label=f'Train ({ep} ep)', color=colors[i], linestyle='-')
    axes[0].plot(range(1, len(h['val_accuracy']) + 1), h['val_accuracy'], label=f'Val ({ep} ep)', color=colors[i], linestyle='--')

axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(loc='lower right')

for i, ep in enumerate(epoch_list):
    h = history_results[ep]
    # พล็อต Loss
    axes[1].plot(range(1, len(h['loss']) + 1), h['loss'], label=f'Train ({ep} ep)', color=colors[i], linestyle='-')
    axes[1].plot(range(1, len(h['val_loss']) + 1), h['val_loss'], label=f'Val ({ep} ep)', color=colors[i], linestyle='--')

axes[1].set_title('Model Loss Comparison')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='upper right')

fig.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "epochs_line_comparison.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)
print(f"\nSaved comparison line chart to: {save_path}")