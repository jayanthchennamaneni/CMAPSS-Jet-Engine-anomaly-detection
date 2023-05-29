from train import RegressionModel, model
from fastapi import FastAPI
import uvicorn
import torch
import numpy as np

app = FastAPI()

# Load the trained model
model = RegressionModel(26)
model = torch.load(r"./models/model.pt")

@app.get("/")
async def root():
    return {"this is home"}

@app.post("/predict")
async def predict(data: list):
    data = np.array(data).astype('float32')  # convert list to numpy array
    data = torch.from_numpy(data)  # convert numpy array to PyTorch tensor
    with torch.no_grad():
        model.eval()
        prediction = model(data).tolist()  # making prediction
    return {"prediction": prediction}
