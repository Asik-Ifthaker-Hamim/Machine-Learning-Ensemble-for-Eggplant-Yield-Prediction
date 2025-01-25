from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the model and scaler using joblib
model = joblib.load('best_voting_model.pkl')  # Load the trained model
scaler = joblib.load('scaler.pkl')  # Load the scaler used during training

app = Flask(__name__)

# Welcome page route
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Research Highlights page route
@app.route('/research')
def research():
    return render_template('research.html')

# Detailed Description page route
@app.route('/description')
def description():
    return render_template('description.html')

# Prediction form route
@app.route('/predict_form')
def predict_form():
    return render_template('home.html')

# Prediction result route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float
        fw = float(request.form['fw'])  # Fruit Weight
        ph = float(request.form['ph'])  # Plant Height
        tf = float(request.form['tf'])  # Total Fruits

        # Create a NumPy array
        arr = np.array([[fw, ph, tf]], dtype=np.float32)

        # Scale the input data using the same scaler used during training
        arr_scaled = scaler.transform(arr)

        # Make prediction
        pred = model.predict(arr_scaled)

        # Format the prediction to 2 decimal places
        pred_rounded = round(pred[0], 2)  # Round to 2 decimal places

        # Render the result page with the formatted prediction
        return render_template('after.html', data=pred_rounded)
    except Exception as e:
        # Handle errors gracefully
        return f"Error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)