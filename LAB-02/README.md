# 📊 LAB 2: Data Preprocessing (Machine Learning)

## 📋 Objective
This lab focuses on the complete Data Preprocessing pipeline using Python, Pandas, Matplotlib, Seaborn, and Scikit-learn. The goal is to explore raw data, visualize underlying distributions, clean anomalies, and prepare features for Machine Learning model training.

---

## 📁 File Structure

LAB02/
├── LAB-02.py               # Main Python execution script
├ # LAB 2: Data Visualization
Conv── openpowerlifting.csv    # Raw Dataset
└── README.md               # Lab documentation & code breakdown

## How to Run 
1. Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:
- command
pip install pandas numpy matplotlib seaborn scikit-learn

## Comprehensive Code Overview & Explanation
   # LAB 1: Dataset Exploration
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

   erts numerical metrics into visual representations to identify distributions and relationships.

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

   # LAB 3: Data Cleaning
Refines data quality by handling missing values, removing duplicates, fixing logical errors, and converting data types.

1. Missing Value Handling
```python
# Impute numerical missing values using Median
for col in numeric_cols:
    if df_cleaned[col].isnull().sum() > 0:
        col_median = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(col_median)

# Impute categorical missing values using Mode
for col in categorical_cols:
    if df_cleaned[col].isnull().sum() > 0:
        col_mode = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(col_mode)

Why Median? Robust against skewed distributions and severe outliers.
Why Mode? Fills categorical gaps using the most frequent occurrence.

    2. Duplicate Removal
```python
df_cleaned = df_cleaned.drop_duplicates()

Removes redundant identical rows to eliminate bias during model training.
    3. Incorrect Data Correction
```python
for col in numeric_cols:
    incorrect_mask = df_cleaned[col] < 0
    if incorrect_mask.sum() > 0:
        df_cleaned.loc[incorrect_mask, col] = df_cleaned[col].median()
```
Detects impossible physical values (e.g., negative age or negative body weight < 0) and replaces them with median values.

# LAB 4: Feature Engineering
Converts categorical text attributes into machine-readable numeric formats (Encoding).

1. Label Encoding
```python
le = LabelEncoder()
df_cleaned['Division_Encoded'] = le.fit_transform(df_cleaned['Division'])
Assigns a unique integer ID to each string label.

Suitable for binary categories or features with ordinal relationship.

2. High-Cardinality Handling & One-Hot Encoding
```python
# Select top 30 categories and group the rest into 'Other'
top_divisions = df_cleaned['Division'].value_counts().index[:30]
df_cleaned['Division'] = df_cleaned['Division'].apply(lambda x: x if x in top_divisions else 'Other')
# Perform One-Hot Encoding
df_final = pd.get_dummies(df_cleaned, columns=['Division'], prefix='Div', dtype=in
    - High-Cardinality Management: Prevents "Curse of Dimensionality" by grouping sparse classes outside the top 30 into an 'Other' category.One-Hot Encoding: Creates binary vector columns ($0$ or $1$) for nominal categories so algorithms won't assume non-existent mathematical order.
```

## Learning Outcomes
Exploratory Data Analysis (EDA): Mastery of structural diagnostics using Pandas.

Visual Analytics: Creating and reading statistical distributions and feature relationships using Seaborn.

Data Hygiene: Advanced missing-value imputation, constraint checks, and memory management.

Feature Encoding: Handling high-cardinality nominal variables safely using hybrid Label & One-Hot Encoding.