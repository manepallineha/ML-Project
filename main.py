from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        year = int(request.form['year'])
        kms = int(request.form['kms'])
        condition = int(request.form['condition'])

        # Model input
        input_data = np.array([[year, kms, condition]])

        # Prediction
        prediction = model.predict(input_data)

        predicted_price = round(prediction[0], 2)

        return render_template(
            'index.html',
            prediction_text=f"Estimated Car Price: ₹ {predicted_price}"
        )

    except Exception as e:

        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)