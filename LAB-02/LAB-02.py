# import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
        print(f"-> คอลัมน์ '{col}' พบค่าว่าง {missing_count} จุด: เติมเต็มด้วยค่า Median ({col_median:.2f}) เรียบร้อย")
categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
for col in categorical_cols:
    missing_count = df_cleaned[col].isnull().sum()
    if missing_count > 0:
        col_mode = df_cleaned[col].mode()[0]
        df_cleaned[col] = df_cleaned[col].fillna(col_mode)
        print(f"-> คอลัมน์ '{col}' พบค่าว่าง {missing_count} จุด: เติมเต็มด้วยค่า Mode ('{col_mode}') เรียบร้อย")
# - Duplicate Removal