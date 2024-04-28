import pandas as pd
import joblib

# Load the saved models
rf_model = joblib.load('Backend/Joblib/rf_model.joblib')
gb_model = joblib.load('Backend/Joblib/gb_model.joblib')
et_model = joblib.load('Backend/Joblib/et_model.joblib')
voting_model = joblib.load('Backend/Joblib/voting_model.joblib')

# Load the CSV file to calculate HF value
housing_data = pd.read_csv('Backend/housing_data.csv')

# Define the input data
input_data = {}

input_data['year'] = int(input("Enter the year: ") or 2025)
input_data['location'] = input("Enter the location: ") or "Pal"
input_data['sqft'] = int(input("Enter the square footage: ") or 2000)
input_data['bedrooms'] = int(input("Enter the number of bedrooms: ") or 4)
input_data['bathrooms'] = int(input("Enter the number of bathrooms: ") or 3)
input_data['amenities'] = float(input("Enter the amenities score: ") or 7.5)
input_data['real_estate_type'] = input("Enter the real estate type: ") or "House"


# Get area rank from location
areas_data = {
    "Area1": {"ranking": 5, "locations": ["Adajan", "Pal", "Vesu"]},
    "Area2": {"ranking": 4, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
    "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
    "Area4": {"ranking": 2, "locations": ["Sarthana", "Katargam", "Udhna"]},
    "Area5": {"ranking": 1, "locations": ["Sachin", "Dindoli", "Bhestan"]}
}

location = input_data['location']
area_rank = None
for area, details in areas_data.items():
    if location in details['locations']:
        area_rank = details['ranking']
        break

# Calculate HF value
year_range = housing_data['year'].max() - housing_data['year'].min()
hf_slope = (housing_data['HF'].max() - housing_data['HF'].min()) / year_range
input_data['HF'] = hf_slope * (input_data['year'] - housing_data['year'].min()) + housing_data['HF'].min()

# Update location and real_estate_type to integer values
input_data['location'] = area_rank
real_estate_type_priority = {
    "Apartment": 1,
    "House": 2,
    "Villa": 3
}
input_data['real_estate_type'] = real_estate_type_priority[input_data['real_estate_type']]

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Use the models to predict the price
rf_prediction = rf_model.predict(input_df)[0].round(2)
gb_prediction = gb_model.predict(input_df)[0].round(2)
et_prediction = et_model.predict(input_df)[0].round(2)
voting_prediction = voting_model.predict(input_df)[0].round(2)

# Print the predictions
print(f"Random Forest Prediction: {rf_prediction}")
print(f"Gradient Boosting Prediction: {gb_prediction}")
print(f"Extra Trees Prediction: {et_prediction}")
print(f"Voting Regressor Prediction: {voting_prediction}")
