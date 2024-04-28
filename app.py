from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

rf_prediction = 0
gb_prediction = 0
et_prediction = 0
voting_prediction = 0

# Load the saved models
rf_model = joblib.load('Backend/Joblib/rf_model.joblib')
gb_model = joblib.load('Backend/Joblib/gb_model.joblib')
et_model = joblib.load('Backend/Joblib/et_model.joblib')
voting_model = joblib.load('Backend/Joblib/voting_model.joblib')

# Load the CSV file to calculate HF value
housing_data = pd.read_csv('Backend/housing_data.csv')

# Get area rank from location
areas_data = {
    "Area1": {"ranking": 5, "locations": ["Adajan", "Pal", "Vesu"]},
    "Area2": {"ranking": 4, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
    "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
    "Area4": {"ranking": 2, "locations": ["Sarthana", "Katargam", "Udhna"]},
    "Area5": {"ranking": 1, "locations": ["Sachin", "Dindoli", "Bhestan"]}
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    input_data = {}

    input_data['year'] = int(request.form.get('year', 2025))
    input_data['location'] = request.form.get('location', 'Pal')
    input_data['sqft'] = int(request.form.get('sqft', 2000))
    input_data['bedrooms'] = int(request.form.get('bedrooms', 4))
    input_data['bathrooms'] = int(request.form.get('bathrooms', 3))
    input_data['amenities'] = float(request.form.get('amenities', 7.5))
    input_data['real_estate_type'] = request.form.get('real_estate_type', 'House')

    location = input_data['location']
    area_rank = None
    for area, details in areas_data.items():
        if location in details['locations']:
            area_rank = details['ranking']
            break

    year_range = housing_data['year'].max() - housing_data['year'].min()
    hf_slope = (housing_data['HF'].max() - housing_data['HF'].min()) / year_range
    input_data['HF'] = hf_slope * (input_data['year'] - housing_data['year'].min()) + housing_data['HF'].min()

    input_data['location'] = area_rank
    real_estate_type_priority = {
        "Apartment": 1,
        "House": 2,
        "Villa": 3
    }
    input_data['real_estate_type'] = real_estate_type_priority.get(input_data['real_estate_type'], 2)

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Handle missing values (NaN) in the input data
    input_df.fillna(0, inplace=True)  # Replace NaN with 0, you can use other strategies like mean, median, or dropna

    rf_prediction = rf_model.predict(input_df)[0].round(2)
    gb_prediction = gb_model.predict(input_df)[0].round(2)
    et_prediction = et_model.predict(input_df)[0].round(2)
    voting_prediction = voting_model.predict(input_df)[0].round(2)

    print(f"Random Forest Prediction: {rf_prediction}")
    print(f"Gradient Boosting Prediction: {gb_prediction}")
    print(f"Extra Trees Prediction: {et_prediction}")
    print(f"Voting Regressor Prediction: {voting_prediction}")

    return jsonify({
        'Random Forest Prediction': rf_prediction,
        'Gradient Boosting Prediction': gb_prediction,
        'Extra Trees Prediction': et_prediction,
        'Voting Regressor Prediction': voting_prediction
    })


if __name__ == '__main__':
    app.run(debug=True)
