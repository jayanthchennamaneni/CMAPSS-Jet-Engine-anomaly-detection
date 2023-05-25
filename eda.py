import pandas as pd

# Specify the file path
file_path = r'/Users/cj/Desktop/CMAPSS-Jet-Engine-anomaly-detection/data/train_FD001.txt'

# Read the text file into a pandas DataFrame
df = pd.read_csv(file_path, delimiter=' ', header=None)


# Check for missing values
print(df.isnull().sum())




