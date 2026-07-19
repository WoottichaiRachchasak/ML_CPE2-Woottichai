# import pandas
import pandas as pd
#1.Load Dataset
df = pd.read_csv("openpowerlifting.csv")
#Print the first 5 rows of data in the table.
print(df.head())
#Print the overall structure of the table
print(df.info())
#PrintCheckTableSize
print(df.shape)