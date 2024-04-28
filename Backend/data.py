import pandas as pd
import numpy as np

areas_data = {
    "Area1": {"ranking": 5, "locations": ["Adajan", "Pal", "Vesu"]},
    "Area2": {"ranking": 4, "locations": ["Athwa", "Ghod Dod Road", "City Light"]},
    "Area3": {"ranking": 3, "locations": ["Piplod", "Varachha", "Althan"]},
    "Area4": {"ranking": 2, "locations": ["Sarthana", "Katargam", "Udhna"]},
    "Area5": {"ranking": 1, "locations": ["Sachin", "Dindoli", "Bhestan"]}
}

areas = {area: data["ranking"] for area, data in areas_data.items()}
area_locations = {area: data["locations"] for area, data in areas_data.items()}

# Define price ranges for each area and year (based on the trend)
price_ranges = {
    "Area1": {year: (800 + 30 * (year - 2000), 1200 + 40 * (year - 2000)) for year in range(2000, 2024)},
    "Area2": {year: (600 + 20 * (year - 2000), 900 + 30 * (year - 2000)) for year in range(2000, 2024)},
    "Area3": {year: (700 + 20 * (year - 2000), 1000 + 30 * (year - 2000)) for year in range(2000, 2024)},
    "Area4": {year: (550 + 20 * (year - 2000), 850 + 30 * (year - 2000)) for year in range(2000, 2024)},
    "Area5": {year: (400 + 10 * (year - 2000), 700 + 20 * (year - 2000)) for year in range(2000, 2024)}
}

# Define trends for crime rates
crime_rate_trends = {
    "Area1": np.linspace(8, 2, num=24), 
    "Area2": np.linspace(10, 4, num=24),
    "Area3": np.linspace(12, 6, num=24),
    "Area4": np.linspace(14, 8, num=24),
    "Area5": np.linspace(16, 10, num=24)
}

# Function to classify areas
def classify_area(location):
    for area, locations in area_locations.items():
        if location in locations:
            return area
    return "Other"

# Generate sample data
data = []
for year in range(2000, 2024):
    for _ in range(1000):  # Generate 1000 records for each year
        area = np.random.choice(list(areas.keys()))
        location = np.random.choice(area_locations[area])
        sqft = np.random.randint(500, 3001)
        crime_rates = crime_rate_trends[area][year - 2000]
        bedrooms = np.random.randint(1, 6)
        bathrooms = np.random.randint(1, 4)
        amenities = np.random.randint(1, 11)
        real_estate_type = np.random.choice(['Apartment', 'House', 'Villa'])
        data.append([year, location, sqft, bedrooms, bathrooms, amenities, real_estate_type, crime_rates, area])

# Create DataFrame
df = pd.DataFrame(data, columns=['year', 'location', 'sqft', 'bedrooms', 'bathrooms', 'amenities', 'real_estate_type', 'crime_rates', 'area'])


# Merge the DataFrame with the provided data
data = {
    'year': [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000],
    'Population': [8331000, 8065000, 7784000, 7490000, 7185000, 6874000, 6564000, 6251000, 5954000, 5671000, 5401000, 5144000, 4900000, 4667000, 4445000, 4233000, 4032000, 3840000, 3658000, 3484000, 3318000, 3160000, 3010000, 2867000, 2867000],
    'GDP (in crores)': [25630, 22620, 19440, 16590, 16500, 15030, 13290, 11670, 10290, 8080, 7240, 6160, 455, 365, 330, 228, 213, 189, 6, 153, 149, 122, 113, 107, 107],
    'Fixed Deposit Interest Rate (% p.a.)': [5.35, 5.90, 5.35, 5.35, 5.35, 5.70, 6.25, 6.25, 6.50, 7.00, 8.50, 8.75, 8.75, 9.00, 8.75, 6.50, 7.75, 7.50, 7.75, 6.25, 5.75, 5.25, 5.50, 8.50, 9.50],
    'Literacy Rate (%)': [87.89, 87.30, 86.70, 86.15, 85.65, 85.20, 84.80, 84.45, 84.15, 83.90, 83.70, 83.55, 83.45, 83.40, 83.40, 83.40, 83.30, 83.00, 82.50, 81.80, 80.90, 79.80, 78.50, 77.00, 75.00]
}

new_data = pd.DataFrame(data)

# Merge the new data with the existing DataFrame based on 'year'
df = df.merge(new_data, on='year', how='left')

# Interpolate missing Literacy Rate values
df['Literacy Rate (%)'] = df['Literacy Rate (%)'].interpolate()

# Interpolate missing GDP values
df['GDP (in crores)'] = df['GDP (in crores)'].interpolate()

# Interpolate missing Fixed Deposit Interest Rate values
df['Fixed Deposit Interest Rate (% p.a.)'] = df['Fixed Deposit Interest Rate (% p.a.)'].interpolate()

# Interpolate missing Population values
df['Population'] = df['Population'].interpolate()

# Define the parameters for the exponential function
a = 0.2  # Rate of increase
b = 0.1  # Offset

# Calculate the development rate for each year using the exponential function
df['Development Rate'] = np.exp(a * (df['year'] - 2000)) + b


# Round all numerical columns to 2 decimal places
df = df.round(2)

# Calculate HF
df['HF'] = ((df['Population']/1000000) * (df['GDP (in crores)']/10) * df['Literacy Rate (%)'] * df['Development Rate']) / (df['crime_rates'] * df['Fixed Deposit Interest Rate (% p.a.)'])

# Round HF to 2 decimal places
df['HF'] = df['HF'].round(2)

# Define weights for real estate types
real_estate_type_weights = {
    'Villa': 0.010,
    'House': 0.005,
    'Apartment': 0.001
}

# Define approximate prices for bedrooms and bathrooms
price_per_bedroom = 1000000
price_per_bathroom = 500000

# Map the area-wise price based on location and year
df['Area-wise Price'] = df.apply(lambda row: price_ranges[row['area']][row['year']][0], axis=1)

# Calculate the price
df['price'] = (((df['Area-wise Price'] * df['sqft']) * df['real_estate_type'].map(real_estate_type_weights)) + ((price_per_bedroom * df['bedrooms']) + (price_per_bathroom * df['bathrooms']))) * df['HF']*0.001 * (df['amenities']*0.0001)
# Round price to 2 decimal places
df['price'] = df['price'].round(2)

# Save the updated DataFrame to the CSV file
df.to_csv('Backend/housing_data.csv', index=False)

print(df.head())
print(df.tail())

print(df.shape)

