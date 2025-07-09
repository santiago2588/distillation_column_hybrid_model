
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
file_path = '/home/ubuntu/upload/DistillationColumnDataset.xlsx'
df = pd.read_excel(file_path)

# Define features (X) and targets (y)
# Exclude 'Time', 'MoleFractionTX', and 'MoleFractionHX' from features
X = df.drop(['Time', 'MoleFractionTX', 'MoleFractionHX'], axis=1)
y = df[['MoleFractionTX', 'MoleFractionHX']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the trained model
joblib.dump(rf_model, '/home/ubuntu/random_forest_model.joblib')
print('Random Forest model trained and saved as random_forest_model.joblib')

# Save feature names for later use in the notebook
with open('/home/ubuntu/rf_model_features.txt', 'w') as f:
    for feature in X.columns:
        f.write(f'{feature}\n')
print('Feature names saved to rf_model_features.txt')


