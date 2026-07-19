# import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
#LAB_01 Dataset Exploration
# - Load Dataset
df = pd.read_csv("openpowerlifting.csv")
# - Display Shapee
print("------------------------------------------------------------------------------------------------------")
print("Shape of the dataset:",df.shape)
print("------------------------------------------------------------------------------------------------------")
# - Display Data Types

print("Data Types of the dataset:\n",df.dtypes)

# - Display Summary Statistics
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Summary Statistics of the dataset:\n",df.describe())
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
# - Display Missing Values
print("Missing Values in the dataset:\n",df.isnull().sum())
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
# - Display Duplicate Records
print("Duplicate Records in the dataset:\n",df.duplicated().sum())
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
# - Display Class Distribution
print("Class Distribution in the dataset:\n",df["BestSquatKg"].value_counts())
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
#LAB_02 Data Visualization
# - Histogram
sns.histplot(data=df, x='Age', kde=True, bins=30)
plt.title('Age Distribution of Athletes')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
# - Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.draw()
plt.show()
#LAB_03 Data Cleaning
# - Missing Value Handling
print("\n[ 1. Missing Value Handling ]")

df_cleaned = df.copy()

numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        col_median = df_cleaned[col].median()
        # เปลี่ยนมาใช้ df_cleaned แก้ไขค่าตรงๆ แบบนี้
        df_cleaned[col] = df_cleaned[col].fillna(col_median)
        print(f"-> colum '{col}' missing count: {missing_count} Filled with median value ({col_median:.2f})")
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        col_mode = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(col_mode)
        print(f"colum '{col}' missing count: {missing_count} : Filled with mode value ('{col_mode}')")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
# - Duplicate Removal
rows_before = df_cleaned.shape[0]
df_cleaned = df_cleaned.drop_duplicates()
rows_after = df_cleaned.shape[0]
print(f"before delete: {rows_before} row | after delete: {rows_after} row")
print(f"Deleted data: {rows_before - rows_after} rows")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
# - Incorrect Data Correction
for col in numeric_cols:
    incorrect_mask = df_cleaned[col] < 0
    incorrect_count = incorrect_mask.sum()
    if incorrect_count > 0:
        col_median = df_cleaned[col].median()
        df_cleaned.loc[incorrect_mask, col] = col_median
        print(f"-> Detected {incorrect_count} erroneous negative values ​​in column '{col}': corrected to the median value")
    else:
        print(f"-> Column '{col}': No anomalous negative values ​​found")
# - Data Type Conversion
if 'Age' in df_cleaned.columns:
    df_cleaned['Age'] = df_cleaned['Age'].round().astype('int64')
    print("-> Converted column 'Age' from Float to Integer")
    print("-> Converted column 'Age' from Float to Integer ")

for col in categorical_cols:
    df_cleaned[col] = df_cleaned[col].astype('category')
print(f"-> Converted all Object-type columns ({len(categorical_cols)}) to Category type ")
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
# - Compare Mean, Median
print("# - Compare Mean, Median")
mean_age = df_cleaned['Age'].mean()
median_age = df_cleaned['Age'].median()
print(f"Age - Mean: {mean_age:.2f}, Median: {median_age:.2f}")

if 'TotalKg' in df_cleaned.columns:
    mean_total = df_cleaned['TotalKg'].mean()
    median_total = df_cleaned['TotalKg'].median()
    print(f"TotalKg - Mean: {mean_total:.2f}, Median: {median_total:.2f}")
print("-" * 50)
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
#LAB_04 Feature Engineering
#Label Encoding 
le = LabelEncoder()
if 'Division' in df_cleaned.columns:
    df_cleaned['Division_Encoded'] = le.fit_transform(df_cleaned['Division'])
    print("-> Label Encoding Completed for 'Division'")
    print("   Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
# 2. One-Hot Encoding 
print("--- เริ่มจัดการปัญหา High Cardinality ในคอลัมน์ Division ---")

top_divisions = df_cleaned['Division'].value_counts().index[:30]

df_cleaned['Division'] = df_cleaned['Division'].apply(lambda x: x if x in top_divisions else 'Other')

print(f"ลดรูป Division เหลือทั้งหมด: {df_cleaned['Division'].nunique()} กลุ่ม (รวม 'Other')")

df_final = pd.get_dummies(df_cleaned, columns=['Division'], prefix='Div', dtype=int)

print("--- ทำ One-Hot Encoding สำเร็จแล้ว! ---")
print(f"ขนาดของ DataFrame ใหม่: {df_final.shape}")