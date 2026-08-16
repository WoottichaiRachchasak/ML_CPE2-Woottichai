# Machine Learning Lab_04: Clustering Model (Wine Quality Dataset)

## Directory Structure & File Overview
### **`data_loader.py`**
- Loads 11 chemical features from `WineQT.csv`.
- mits target label processing to perform unsupervised learning.
- pplies `StandardScaler` across all features ($Mean=0, SD=1$).
- Returns both scaled data (`X`) for tensor calculations and raw data (`X_raw`) for cluster interpretation.

### **`kmeans_tf.py`**
- Implements the `TFKMeans` class built on TensorFlow.
- Randomly initializes centroids and iteratively updates cluster centers until convergence.
- Computes **Inertia** (Sum of Squared Distances) to assess intra-cluster compactness.

### **`knn_tools.py`**
- Implements the `KNNClusterAssigner` class.
- Accepts new unassigned data points and assigns them to the nearest cluster using k-NN voting based on existing cluster labels.

### **`visualize.py`**
- **`plot_elbow()`**: Generates an Elbow Curve plotting $k$ versus Inertia (`01_elbow.png`).
- **`plot_clusters()`**: Creates a scatter plot visualizing cluster distribution across feature pairs such as `alcohol` and `pH` (`02_clusters.png`).

### **`main.py`**
- Main execution script covering 5 workflow steps:
    1. **Load Data**: Loads and scales chemical feature data.
    2. **Find Best k**: Tests $k = 2$ through $8$, computing Inertia and Silhouette Scores to build the Elbow plot.
    3. **Model Execution**: Fits K-Means with the chosen $k$ and checks silhouette performance.
    4. **Cluster Profiling**: Calculates mean chemical feature values per cluster and exports a summary (`cluster_summary.csv`).
    5. **New Data Assignment**: Simulates predicting clusters for new samples using `KNNClusterAssigner` and verifies accuracy against K-Means labels.

####  Features Used for Clustering

Uses all 11 numerical chemical features:
`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`