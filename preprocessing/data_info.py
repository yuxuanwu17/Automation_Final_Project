import pandas as pd

df = pd.read_csv("data/featureSelectedAllDataWithY.csv")
# df = pd.read_csv("data/Cleaned_dat_encoded.csv")
# df = pd.read_csv("../data/Cleaned_dat.csv")

# Display the shape of the data frame
print('Shape of data frame:', df.shape)

print(df.columns)

# Display the first few rows of the data frame
print('\nFirst few rows of data frame:')
print(df.head())