import pandas as pd

# Specify the file path
file_path = r'/Users/cj/Desktop/CMAPSS-Jet-Engine-anomaly-detection/data/train_FD004.txt'

# Read the text file into a pandas DataFrame
df = pd.read_csv(file_path, delimiter=' ', header=None)

# Drop columns with NaN values
df = df.dropna(axis=1, how='all')

# Check for missing values
df.head(1)





