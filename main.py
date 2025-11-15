from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, encoders, and top features
model = joblib.load('final_model3_top10.pkl')
encoders = joblib.load('encoders_dict.pkl')
top_10_features = joblib.load('top_10_features2.pkl')
top_10_features = list(top_10_features)  # Convert to list in case it's a NumPy array or Series

@app.route('/')
def home():
    dropdown_options = {}

    for feature in top_10_features:
        if feature in encoders:
            dropdown_options[feature] = list(encoders[feature].classes_)

    return render_template('food5.html', top_10_features=top_10_features, dropdown_options=dropdown_options)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        features = []
        for feature_name in top_10_features:
            value = request.form.get(feature_name)
            features.append(value)

        # Encode categorical variables
        for feature_name, encoder in encoders.items():
            if feature_name in top_10_features:
                index = top_10_features.index(feature_name)
                features[index] = encoder.transform([features[index]])[0]

        # Convert to numpy array
        features = np.array(features).reshape(1, -1)

        # Predict log price
        log_price = model.predict(features)[0]

        # Convert back to actual price
        actual_price = np.exp(log_price)

        return jsonify({'prediction': actual_price})


if __name__ == '__main__':
    app.run(debug=True)
