# Breast Cancer Prediction Flask App

This project is a web application that predicts breast cancer using a machine learning model built with logistic regression. The application takes in key clinical features and predicts whether a sample is likely "Positive" (indicating cancer) or "Negative" (no cancer).

## Features

- **Machine Learning Model**: Logistic regression model trained on the Breast Cancer Wisconsin dataset.
- **Web Interface**: Simple, user-friendly HTML form for inputting feature data.
- **Instant Prediction**: Displays prediction results immediately after form submission.

## Files

- `app.py`: The Flask application file that serves the web interface and handles predictions.
- `templates/index.html`: The HTML file for the user interface, allowing users to input data for prediction.
- `BREAST CANCER PREDICTION _ML.ipynb`: Jupyter notebook used for data analysis and model training.

## Selected Features for Prediction

The app uses 10 important features from the dataset:

1. Mean Radius
2. Mean Texture
3. Mean Perimeter
4. Mean Area
5. Mean Smoothness
6. Mean Compactness
7. Mean Concavity
8. Mean Concave Points
9. Mean Symmetry
10. Mean Fractal Dimension

## Prerequisites

- Python 3.x
- Flask
- scikit-learn
- pandas

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/koushik-yarra/breast-cancer-prediction-flask-app.git
    cd breast-cancer-prediction-flask-app
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    **Note**: If `requirements.txt` is not provided, install Flask and scikit-learn manually:
    ```bash
    pip install flask scikit-learn pandas
    ```

3. **Run the Application**:
    ```bash
    python app.py
    ```

4. **Access the Web App**:
    Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage

1. Enter values for each of the 10 clinical features in the form provided on the web page.
2. Click the "Predict" button.
3. The application will display the prediction as either "Positive" (indicating cancer) or "Negative" (no cancer).


