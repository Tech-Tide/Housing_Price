import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('Backend/housing_data.csv')

# Convert location to integer based on areas_data
areas_data = {
    "Area1": {"ranking": 5, "locations": ["Adajan", "Pal", "Vesu"]},
    "Area2": {"ranking": 4, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
    "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
    "Area4": {"ranking": 2, "locations": ["Sarthana", "Katargam", "Udhna"]},
    "Area5": {"ranking": 1, "locations": ["Sachin", "Dindoli", "Bhestan"]}
}

location_to_int = {}
for area, details in areas_data.items():
    for location in details['locations']:
        location_to_int[location] = details['ranking']

data['location'] = data['location'].map(location_to_int)

# Map 'real_estate_type' to integers
type_priority = {
    "Apartment": 1,
    "House": 2,
    "Villa": 3
}

data['real_estate_type'] = data['real_estate_type'].map(type_priority)

# Select the specified features and target variable
features = ['year', 'location', 'sqft', 'bedrooms', 'bathrooms', 'amenities', 'real_estate_type', 'HF']
X = data[features]
y = data['price']

# Assign feature priority values
feature_priority = {
    "sqft": 1,
    "location": 2,
    "amenities": 3,
    "bedrooms": 4,
    "real_estate_type": 5,
    "bathrooms": 6,
    "HF": 7,
    "year": 8
}

# Sort features based on priority values
sorted_features = sorted(features, key=lambda x: feature_priority[x])

# Train the models
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
et_model = ExtraTreesRegressor(random_state=42)

# Fit the models
rf_model.fit(X, y)
gb_model.fit(X, y)
et_model.fit(X, y)

# Combine the models using VotingRegressor
voting_model = VotingRegressor([('rf', rf_model), ('gb', gb_model), ('et', et_model)])
voting_model.fit(X, y)

# Save the models
joblib.dump(rf_model, 'Backend/Joblib/rf_model.joblib')
joblib.dump(gb_model, 'Backend/Joblib/gb_model.joblib')
joblib.dump(et_model, 'Backend/Joblib/et_model.joblib')
joblib.dump(voting_model, 'Backend/Joblib/voting_model.joblib')

print("Models trained and saved successfully.")

# Load the test dataset
test_data = pd.read_csv('Backend/test.csv')

# Convert location to integer for test data
test_data['location'] = test_data['location'].map(location_to_int)

# Map 'real_estate_type' to integers for test data
test_data['real_estate_type'] = test_data['real_estate_type'].map(type_priority)

# Select the specified features and target variable for test data
X_test = test_data[features]
y_test = test_data['price']

# Calculate mean R-squared value across all models
rf_r2 = rf_model.score(X_test, y_test)
gb_r2 = gb_model.score(X_test, y_test)
et_r2 = et_model.score(X_test, y_test)
voting_r2 = voting_model.score(X_test, y_test)

print(rf_r2)
print(gb_r2)
print(et_r2)
print(voting_r2)

# Calculate the mean R-squared value
mean_r2 = (rf_r2 + gb_r2 + et_r2 + voting_r2) / 4

# Calculate the percentage accuracy
accuracy_percentage = mean_r2 * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")


import matplotlib.pyplot as plt

# Plotting R-squared values
models = ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'Voting']
r2_values = [rf_r2, gb_r2, et_r2, voting_r2]



# Plotting feature importance (example for RandomForestRegressor)
feature_importance = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='lightcoral')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to show most important features on top
plt.show()
