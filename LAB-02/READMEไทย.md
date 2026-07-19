# LAB_01 Dataset Exploration
- Load Dataset: Load the dataset file `openpowerlifting.csv` into the system using Pandas.
- Display Shape: Check the dataset dimensions to see the total number of rows and columns.
- Display Data Types: Check the type of variables (Data Types) in each column.
- Display Summary Statistics: Retrieve basic descriptive statistics such as Mean, Median, Maximum, and Minimum values.
- Display Missing: Check the number of missing values.
- Duplicate Check: Check for duplicate data rows.
- Display Class Distribution: View the distribution of the selected column, such as `BestSquatKg`.

# LAB_02 Data Visualization
- Histogram: Create a frequency histogram with a curve to view the distribution of the selected column.
- Correlation Heatmap: Create a heatmap to analyze relationships and correlation coefficients.

# LAB_03 Data Cleaning
- Missing Value Handling 
    - Numeric columns: Replace missing values with the median of that column.
    - Categorical columns: Replace missing values with the mode or the most frequent value.
- Duplicate Removal: Delete duplicate data rows from the dataset.
- Incorrect Data Correction: Check for logically incorrect data, such as negative values in numeric columns. When found, replace them with the median instead.
- Data Type Conversion: Adjust the `Age` column from float to integer.
    * Convert all text columns to categorical variables to save memory space.
- Compare mean/median: Comparison of mean and median values from the calculated comparison of `Age` and `TotalKg` variables to summarize behavior trends.

# LAB_04 Feature Engineering
- Label Encoding: Convert the `Division` column into numbers $0, 1, 2, ..., 4245$ in a new column named `Division_Encoded` to help save storage space.
- One-Hot Encoding & Memory Error Mitigation
    * Problem: The `Division` column has too many unique sub-models (4,246 forms). If standard One-Hot Encoding is performed directly, it will consume up to 12.2 GB of RAM until the system crashes.
    * Solution: Filter and select only the Top 30 most popular models, while all remaining models are grouped together into the 'Other' group.
    * Result: Reduce the groups to only 31 groups, then successfully perform `pd.get_dummies()` quickly without running out of RAM, and the data is ready to be used for training the model immediately.