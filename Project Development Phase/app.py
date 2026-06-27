
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('HDI.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    life_expectancy = float(request.form['life_expectancy'])
    expected_schooling = float(request.form['expected_schooling'])
    mean_schooling = float(request.form['mean_schooling'])
    gni_per_capita = float(request.form['gni_per_capita'])

    features = np.array([[life_expectancy, expected_schooling, 
                          mean_schooling, gni_per_capita]])
    prediction = model.predict(features)[0]

    if prediction >= 0.8:
        category = "Very High"
    elif prediction >= 0.7:
        category = "High"
    elif prediction >= 0.55:
        category = "Medium"
    else:
        category = "Low"

    return render_template('resultnew.html', 
                          hdi_score=round(prediction, 3),
                          hdi_category=category)

if __name__ == "__main__":
    app.run(debug=True)
