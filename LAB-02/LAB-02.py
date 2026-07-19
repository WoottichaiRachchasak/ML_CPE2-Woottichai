# import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#LAB 01 Dataset Exploration
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
print("Class Distribution in the dataset:\n",df["Best3SquatKg"].value_counts())
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
#LAB 02 Data Visualization