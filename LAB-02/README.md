# 📊 LAB 2: Data Preprocessing (Machine Learning)

## 📋 Objective
This lab focuses on the complete Data Preprocessing pipeline using Python, Pandas, Matplotlib, Seaborn, and Scikit-learn. The goal is to explore raw data, visualize underlying distributions, clean anomalies, and prepare features for Machine Learning model training.

---

## 📁 File Structure

LAB02/
├── LAB-02.py               # Main Python execution script
├── openpowerlifting.csv    # Raw Dataset
└── README.md               # Lab documentation & code breakdown

## How to Run 
1. Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:
- command
pip install pandas numpy matplotlib seaborn scikit-learn

# Comprehensive Code Overview & Explanation
     LAB 1: Dataset Exploration
Examines the initial specification, structure, and quality of the raw dataset.
- Load Dataset
        pd.read_csv("openpowerlifting.csv") — Loads the dataset into a Pandas DataFrame.
- Display Shapee
        df.shape — Displays total rows and columns.
- Display Data Types
        df.dtypes — Inspects data types for each feature (float64, object, etc.).
- Display Summary Statistics
        df.describe() — Computes summary statistics (Mean, Standard Deviation, Min/Max, Quartiles).
- Display Missing Values
        df.isnull().sum() — Detects missing/null values (NaN) per column.
- Display Duplicate Records
        df.duplicated().sum() — Counts fully duplicate rows across the dataset.
- Display Class Distribution
        df["BestSquatKg"].value_counts() — Analyzes class/value distribution for key performance target columns.

## LAB 2: Data Visualization
Converts numerical metrics into visual representations to identify distributions and relationships.

1. Age Distribution Histogram (sns.histplot)
What it does: Plots the age frequency of athletes with a Kernel Density Estimate (kde=True) overlay.

How to interpret:

X-axis: Athlete Age range.

Y-axis: Athlete Count.

Helps detect skewness, modality, and age-related outliers in the dataset.

2. Correlation Heatmap (sns.heatmap)
What it does: Calculates pairwise Pearson correlation coefficients (df.corr()) across numeric columns and displays them in a color-coded matrix.

How to interpret:

Values range from -1.0 to +1.0

Red / Warm colors: Strong positive correlation (features increase together).

Blue / Cool colors: Strong negative correlation (one feature increases as the other decreases).

Near 0.0: No linear relationship between features.