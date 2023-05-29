from train import RegressionModel
import torch
import pandas as pd
import numpy as np

test_column_names = ['unit_number', 'time', 'operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + ['sensor_measurement_{}'.format(i) for i in range(1, 22)]

df1 = pd.read_csv(r'./data/test_FD004.txt', delimiter=' ', header=None)
df1.drop(df1.columns[[26, 27]], axis=1, inplace=True)  # drop the last two columns, which are NaNs
df1.columns = test_column_names

# Group the test data by engine id, and take the last row from each group
X_test_last_cycle = df1.groupby('unit_number').last().reset_index()

# Convert the dataframe to PyTorch tensor
X_test_tensor = torch.tensor(X_test_last_cycle.values, dtype=torch.float)

# Load the model
model = RegressionModel(26)
model = torch.load("./models/model.pt")  # Load the trained weights

model.eval()
with torch.no_grad():
    y_pred1 = model(X_test_tensor)

# Convert the predictions to a numpy array
y_pred1 = y_pred1.squeeze().detach().numpy()


# Load the true RUL values
true_rul = pd.read_csv(r'./data/RUL_FD004.txt', sep=" ", header=None)
true_rul = true_rul[0].values

# Calculate Mean Absolute Error
mae = np.abs(y_pred1 - true_rul).mean()
print(f'Mean Absolute Error on test set: {mae}')
