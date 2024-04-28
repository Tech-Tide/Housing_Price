import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from sklearn.preprocessing import OneHotEncoder


# Load the dataset (assuming 'data' is the DataFrame containing the real estate data)
data = pd.read_csv('Backend/housing_data.csv')

# Define the mapping for areas
areas_data = {
    "Area1": {"ranking": 5, "locations": ["Adajan", "Pal", "Vesu"]},
    "Area2": {"ranking": 4, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
    "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
    "Area4": {"ranking": 2, "locations": ["Sarthana", "Katargam", "Udhna"]},
    "Area5": {"ranking": 1, "locations": ["Sachin", "Dindoli", "Bhestan"]}
}

# Map 'area' to rankings in the dataset
data['area_ranking'] = data['area'].apply(lambda x: areas_data[x]['ranking'])

# Select features for regression and classification tasks
regression_features = ['sqft', 'bedrooms', 'bathrooms', 'crime_rates', 'area_ranking', 'Population', 'GDP (in crores)', 'Fixed Deposit Interest Rate (% p.a.)', 'Literacy Rate (%)', 'Development Rate', 'HF', 'Area-wise Price']
classification_features = ['sqft', 'bedrooms', 'bathrooms', 'location', 'amenities', 'crime_rates', 'area_ranking', 'Population', 'GDP (in crores)', 'Fixed Deposit Interest Rate (% p.a.)', 'Literacy Rate (%)', 'Development Rate', 'HF']

# Split the data into training and test sets
X_reg = data[regression_features]
y_reg = data['price']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

X_cls = data[classification_features]
y_cls = data['real_estate_type']
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Regression task: Train a Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_reg_train, y_reg_train)
y_reg_pred = rf_reg.predict(X_reg_test)
regression_mse = mean_squared_error(y_reg_test, y_reg_pred)
print(f"Regression Mean Squared Error: {regression_mse}")

# Apply one-hot encoding to the 'location' column for classification task
encoder = OneHotEncoder(sparse=False)
X_cls_location_encoded = encoder.fit_transform(X_cls['location'].values.reshape(-1, 1))
X_cls_location_encoded_df = pd.DataFrame(X_cls_location_encoded, columns=encoder.get_feature_names(['location']))

# Concatenate the one-hot encoded 'location' column with the rest of the features
X_cls_encoded = pd.concat([X_cls.drop('location', axis=1), X_cls_location_encoded_df], axis=1)

# Split the data into training and test sets for classification task
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls_encoded, y_cls, test_size=0.2, random_state=42)

# Classification task: Train a Random Forest Classifier with one-hot encoded 'location' column
rf_cls = RandomForestClassifier(random_state=42)
rf_cls.fit(X_cls_train, y_cls_train)
y_cls_pred = rf_cls.predict(X_cls_test)
classification_accuracy = accuracy_score(y_cls_test, y_cls_pred)
print(f"Classification Accuracy: {classification_accuracy}")

# Save the models
joblib.dump(rf_reg, 'rf_regression_model.joblib')
joblib.dump(rf_cls, 'rf_classification_model.joblib')
