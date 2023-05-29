# Predictive Maintenance of Turbofan Engines

The focus of this project is to implement a predictive maintenance model for turbofan engines using Machine Learning. The project takes advantage of a comprehensive dataset made available by NASA, which encompasses different operational settings and conditions of various turbofan engines until failure.

The specific aim of the project is to predict the Remaining Useful Life (RUL) of each engine based on its sensor readings. This is achieved using PyTorch to build and train a regression model capable of forecasting the number of operational cycles an engine has left before it fails.

The developed model is then integrated into a web application served using FastAPI. This web application accepts sensor data and returns a prediction of the RUL.

To enhance the portability and scalability of the application, it is containerized using Docker.This project also implements Continuous Integration (CI) using GitHub Actions, ensuring that the model and the application are constantly updated and tested for performance.

## Directory Structure

The directory structure of this repository is organized as follows:

```
.
├── models              #  This directory stores the trained PyTorch model.
│   └── model.pt
├── Dockerfile          #  Contains the Docker instructions to containerize the FastAPI application
├── eda.py              #  A Python script that performs exploratory data analysis on the turbofan engine dataset.
├── train.py            #  A Python script that trains the regression model.
├── test.py             #  A Python script that test the trained model on test dataset
├── app.py              #  A Python script that hosts the FastAPI  and uses the trained model to make predictions.
└── requirements.txt    #  Contains all the Python dependencies required by the project.

```
## Results

This application was trained and evaluated on four datasets (FD001, FD002, FD003, FD004). The Mean Absolute Error (MAE) between the predicted and actual RUL was used as the evaluation metric. The application achieved the following MAEs on the four test datasets:

1. FD001: ...
2. FD002: ...
3. FD003: ...
4. FD004: ...


## Getting Started

To get the project up and running, follow these steps:

1. Clone the repository.
2. Install the necessary Python packages using `pip install -r requirements.txt`.
3. Download the dataset from [Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) and add to root directory.

## Running the Fast API

The FastAPI application is a simple HTTP server that uses the trained model to make predictions. It exposes two routes:

After creating the Docker image, you can run the Fast API. The API consists of two main endpoints:

1. **`/`**: A simple endpoint returning a welcome message.

2. **`/predict`**: A POST route that accepts a list of sensor readings and returns a prediction of the remaining useful life.

To interact with the API, can use any HTTP client like curl, Postman.

Start the Fast API by running the Docker container:
```
docker run -p 8000:8000 <docker-image-name>:<tag>
``` 

Then, you can access the API at `http://localhost:8000`.

## Continuous Integration

In this project, we have employed CI using GitHub Actions. This facilitates the automatic building, testing, and pushing of our Docker container to the specified Docker registry whenever a push to the main branch is made. It aids in ensuring the code is working as expected, and that the Docker container is always up-to-date with the latest changes.

The CI process can be configured according to specific needs in the `.github/workflows/main.yml` file.

## Additional Resources

- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [FAST API Documentation](https://fastapi.tiangolo.com)
- [Docker Documentation](https://docs.docker.com/)
- [Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)


## License

This project is licensed under the MIT License. See the [LICENSE] file for details.