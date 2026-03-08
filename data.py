import pandas as pd

df = pd.read_csv("C:/Users/Anya Kumar/Documents/Epileptic Seizure Recognition/Epileptic_Seizure_Recognition.csv")  # adjust name if different

print(df.shape)          # expect (11500, 179)
print(df.head())
print(df.columns)
print(df['y'].value_counts())  # class distribution
print(df.isnull().sum().sum()) # missing values check
