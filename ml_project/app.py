from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and prepare the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Select the 10 most important features
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension"
]
X = X[selected_features]

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Save the trained model to avoid retraining each time
with open("logistic_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    with open("logistic_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    # Extract features from the form
    input_data = {
        "mean radius": float(request.form['mean_radius']),
        "mean texture": float(request.form['mean_texture']),
        "mean perimeter": float(request.form['mean_perimeter']),
        "mean area": float(request.form['mean_area']),
        "mean smoothness": float(request.form['mean_smoothness']),
        "mean compactness": float(request.form['mean_compactness']),
        "mean concavity": float(request.form['mean_concavity']),
        "mean concave points": float(request.form['mean_concave_points']),
        "mean symmetry": float(request.form['mean_symmetry']),
        "mean fractal dimension": float(request.form['mean_fractal_dimension'])
    }

    # Create a DataFrame for the model input
    input_features = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_features)
    result = "Positive" if prediction[0] == 1 else "Negative"

    return render_template('index.html', prediction_text=f'Cancer Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
