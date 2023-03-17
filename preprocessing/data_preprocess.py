import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import LabelEncoder

np.random.seed(33)

df = pd.read_csv("../data/Cleaned_dat.csv")
# print(df.head())

# convert all the categorical data to label encoded data
cat_columns = df.select_dtypes(['object']).columns

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each 'object' column in the DataFrame
for col in cat_columns:
    df[col] = label_encoder.fit_transform(df[col])

# print(df.head())
df.to_csv("../data/label_encoded_data.csv", index=False)
