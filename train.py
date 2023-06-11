import torch
from torch import nn
from eda import preprocess_data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

X_train, x_val, Y_train, y_val = preprocess_data()

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
joblib.dump(scaler, 'scaler_feat.pkl') 

target_scaler = MinMaxScaler()
Y_train_scaled = pd.DataFrame(target_scaler.fit_transform(Y_train.values.reshape(-1,1)), columns=['RUL'])
y_val_scaled = pd.DataFrame(target_scaler.transform(y_val.values.reshape(-1,1)), columns=['RUL'])
joblib.dump(target_scaler, 'scaler_target.pkl')

class RegressionModel(nn.Module):
    def __init__(self, input_features):
        super(RegressionModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layer(x)

def train_and_validate(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs=1000):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        y_train_pred = model(X_train_tensor)
        train_loss = criterion(y_train_pred.squeeze(), y_train_tensor)
        train_loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_tensor)
            val_loss = criterion(y_val_pred.squeeze(), y_val_tensor)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}')
        
    # Save the model
    torch.save(model.state_dict(), './models/model2.pt')

    return model


model = RegressionModel(26)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if __name__ == "__main__":

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float)
    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float)

    model = train_and_validate(model, criterion, optimizer, X_train_tensor, Y_train_tensor, x_val_tensor, y_val_tensor, num_epochs=1000)


