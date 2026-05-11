import pickle
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained pipeline model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Home route
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        import pandas as pd

        location = request.form.get("location").strip().lower()
        total_sqft = float(request.form.get("total_sqft"))
        bath = float(request.form.get("bath"))
        bhk = int(request.form.get("BHK"))

        data = pd.DataFrame([[location, total_sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'BHK'])

        prediction = model.predict(data)[0]

        result = f"Estimated Price in thousand: ₹ {round(prediction, 2)}"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", result=result)


# Run app
if __name__ == "__main__":
    app.run(debug=True)