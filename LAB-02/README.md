# Machine Learning Laboratory Report

---

## 📊 LAB 01: Dataset Exploration
* **Load Dataset:** Loaded the raw data file `openpowerlifting.csv` into the system using Pandas.
* **Display Shape:** Checked the dataset dimensions to determine the total number of rows and columns.
* **Display Data Types:** Inspected the specific data type of each column.
* **Display Summary Statistics:** Retrieved descriptive statistics such as Mean, Median, Minimum, and Maximum values.
* **Display Missing Values:** Checked for the count of missing values across all features.
* **Duplicate Check:** Verified the total number of duplicate records in the dataset.
* **Display Class Distribution:** Analyzed the data distribution of the selected target column, `BestSquatKg`.

---

## 📈 LAB 02: Data Visualization
* **Histogram:** Created a frequency histogram with a Kernel Density Estimate (KDE) curve to analyze the distribution of the selected column.
* **Correlation Heatmap:** Generated a heatmap to visualize relationships and correlation coefficients between numerical variables.

---

## 🧼 LAB 03: Data Cleaning
* **Missing Value Handling:** 
  * **Numerical Columns:** Imputed missing data using the column's Median.
  * **Categorical Columns:** Imputed missing data using the column's Mode (most frequent value).
* **Duplicate Removal:** Dropped all duplicated rows from the dataset to ensure data integrity.
* **Incorrect Data Correction:** Scanned for logical errors, such as negative values in numerical columns, and replaced them with the Median.
* **Data Type Conversion:** 
  * Converted the `Age` column from Float to Integer.
  * Cast all text columns to Category type to optimize memory usage.
* **Compare Mean/Median:** Compared the Mean and Median of `Age` and `TotalKg` to summarize data behaviors and skewness.

---

## ⚙️ LAB 04: Feature Engineering
* **Label Encoding:** Transformed the `Division` column into numerical labels ($0, 1, 2, ..., 4245$) under a new column named `Division_Encoded` for efficient storage.
* **One-Hot Encoding & Memory Error Mitigation:**
  * **The Problem:** The `Division` column contained too many unique categories (4,246 styles). Performing a standard One-Hot Encoding would consume 12.2 GB of RAM, causing a system crash (Memory Error).
  * **The Solution:** Filtered and retained only the **Top 30** most popular divisions, while grouping all remaining styles into a single category labeled **'Other'**.
  * **The Result:** Reduced the categories down to 31 groups, allowing `pd.get_dummies()` to execute quickly without memory issues. The processed data is now fully prepared for model training.