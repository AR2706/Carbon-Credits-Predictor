import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# =========================================================
# 1. Load Data
# =========================================================
file_path = 'Final Dataset.csv'
try:
    df = pd.read_csv(file_path)
    print("[OK] Data Loaded Successfully.")
except FileNotFoundError:
    print("[ERROR] Error: 'Final Dataset.csv' not found.")
    exit()

# =========================================================
# 2. Strict Filtering (Physics Constraints)
# =========================================================
df = df.dropna(subset=['CO2 EMI PER UNIT', 'DE', 'Total'])
df = df[df['CO2 EMI PER UNIT'] > 0.4]
df = df[df['CO2 EMI PER UNIT'] < 4.0]
df = df[df['DE'] > 0.1]
df = df[df['Total'] > 0]

# =========================================================
# 3. Feature Engineering
# =========================================================
df['Coal_Intensity'] = df['Total'] / df['DE']

def clean_tech(x):
    if not isinstance(x, str): return "Other"
    x = x.lower()
    if "supercritical" in x: return "Supercritical"
    if "subcritical" in x: return "Subcritical"
    return "Other"

df['Tech_Clean'] = df['Tech'].apply(clean_tech)
df['Load_Factor'] = df['DE'] / (df['DC'] + 1e-5)
df['Import_Ratio'] = df['Import'] / (df['Total'] + 1e-5)

features = [
    'Latitude', 'Longitude', 'State Name', 'Region',
    'Tech_Clean', 'Age', 'Coal_Intensity',
    'Import_Ratio', 'Load_Factor'
]
target = 'CO2 EMI PER UNIT'

X = df[features]
y = df[target]

# =========================================================
# 4. SPLIT STRATEGY (80% Train - 10% Test - 10% Validation)
# =========================================================
# Step 1: Split 80% Train vs 20% Temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 2: Split Temp into 50% Test and 50% Validation (which is 10% and 10% of total)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"\n--- Data Split Statistics ---")
print(f"Training Set:   {len(X_train)} rows (80%)")
print(f"Test Set:       {len(X_test)} rows (10%)")
print(f"Validation Set: {len(X_val)} rows (10%) -> Will be exported to CSV")

# =========================================================
# 5. Pipeline Setup
# =========================================================
numeric_features = ['Latitude', 'Longitude', 'Age', 'Coal_Intensity', 'Import_Ratio', 'Load_Factor']
categorical_features = ['State Name', 'Region', 'Tech_Clean']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

ensemble = VotingRegressor(
    estimators=[
        ('hgb', HistGradientBoostingRegressor(max_iter=500, max_depth=12, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)),
        ('et', ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42))
    ],
    weights=[2, 1, 1]
)

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', ensemble)])

# =========================================================
# 6. Training
# =========================================================
print("\nTraining Model on 80% Data...")
model_pipeline.fit(X_train, y_train)

# =========================================================
# 7. EXPORT VALIDATION CSV (The 10% Slice)
# =========================================================
print("\n--- Creating Validation File ---")

# Predict on the 10% Validation Set
y_val_pred = model_pipeline.predict(X_val)

# Create the export dataframe
validation_export = X_val.copy()
validation_export['ACTUAL_CO2'] = y_val
validation_export['PREDICTED_CO2'] = y_val_pred
# Calculate Error for easy spotting
validation_export['ERROR_PCT'] = abs((validation_export['PREDICTED_CO2'] - validation_export['ACTUAL_CO2']) / validation_export['ACTUAL_CO2']) * 100

# Add Plant Name back for human readability (matching via index)
validation_export['Thermal Plant'] = df.loc[X_val.index, 'Thermal Plant']

# Reorder columns for easier reading
cols = ['Thermal Plant', 'State Name', 'ACTUAL_CO2', 'PREDICTED_CO2', 'ERROR_PCT'] + [c for c in validation_export.columns if c not in ['Thermal Plant', 'State Name', 'ACTUAL_CO2', 'PREDICTED_CO2', 'ERROR_PCT']]
validation_export = validation_export[cols]

# Save to CSV
validation_export.to_csv('Validation_Set_Review.csv', index=False)
print("[OK] 'Validation_Set_Review.csv' has been created successfully!")
print("     (Contains actuals vs predictions for the 10% validation split)")

# =========================================================
# 8. Test Set Evaluation
# =========================================================
y_test_pred = model_pipeline.predict(X_test)

test_r2 = r2_score(y_test, y_test_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\n--- Final Test Metrics (Unseen Data) ---")
print(f"R2 Score: {test_r2:.4f}")
print(f"MAPE:     {test_mape:.2%}")
print(f"RMSE:     {test_rmse:.4f}")

# =========================================================
# 9. USER INPUT LOOP
# =========================================================
print("\n*** SYSTEM READY FOR MANUAL INPUT ***")

def predict_custom_point(lat, lon, state):
    # Create a single row of data for the new point
    # We find the nearest neighbor in the training data to fill in the missing technical details
    full_data = X.copy()
    full_data['dist'] = ((full_data['Latitude'] - lat)**2 + (full_data['Longitude'] - lon)**2)**0.5
    nearest_row = full_data.loc[full_data['dist'].idxmin()]

    input_data = pd.DataFrame({
        'Latitude': [lat],
        'Longitude': [lon],
        'State Name': [state],
        'Region': [nearest_row['Region']],
        'Tech_Clean': [nearest_row['Tech_Clean']],
        'Age': [nearest_row['Age']],
        'Coal_Intensity': [nearest_row['Coal_Intensity']],
        'Import_Ratio': [nearest_row['Import_Ratio']],
        'Load_Factor': [nearest_row['Load_Factor']]
    })

    pred = model_pipeline.predict(input_data)[0]
    print(f"\nLocation: {state} ({lat}, {lon})")
    print(f"Nearest Reference Plant: {df.loc[nearest_row.name, 'Thermal Plant']}")
    print(f"Predicted CO2 Emission:  {pred:.4f} / unit")
    print("-" * 30)

while True:
    print("\nEnter coordinates (or type 'exit'):")
    user_in = input("Format [Lat, Long, State]: ")

    if user_in.lower() == 'exit':
        break

    try:
        parts = user_in.split(',')
        lat = float(parts[0])
        lon = float(parts[1])
        state = parts[2].strip()
        predict_custom_point(lat, lon, state)
    except:
        print("[ERROR] Invalid format. Try: 22.36, 82.71, Chhattisgarh")