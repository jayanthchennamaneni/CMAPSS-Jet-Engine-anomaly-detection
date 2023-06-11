from fastapi import FastAPI
from torch import nn
import uvicorn
import torch
import numpy as np

app = FastAPI()

class RegressionModel(nn.Module):
    def __init__(self, input_features):
        super(RegressionModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer(x)
    
# Load the trained model
model = RegressionModel(26)
model = torch.load(r"./models/model.pt")

@app.get("/")
async def root():
    return {"Testing page"}

@app.post("/predict")
async def predict(data: list):
    data = np.array(data).astype('float32')  # convert list to numpy array
    data = torch.from_numpy(data)  # convert numpy array to PyTorch tensor
    with torch.no_grad():
        model.eval()
        prediction = model(data).tolist()  # making prediction
    return {"prediction": prediction}
