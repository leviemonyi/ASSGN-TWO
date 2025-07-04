import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# -------------------------
# Step 1: Create the Dataset
# -------------------------
data = {
    'Number_of_Vehicles': [1, 2, 3, 4, 2, 3, 1, 2, 4, 5],
    'Weather_Condition': [1, 2, 1, 3, 2, 2, 1, 1, 3, 3],  # 1: Clear, 2: Rain, 3: Fog
    'Light_Condition': [1, 2, 1, 2, 1, 2, 1, 1, 2, 2],    # 1: Daylight, 2: Darkness
    'Speed_Limit': [30, 40, 50, 60, 30, 70, 20, 40, 60, 70],
    'Accident_Severity': [1, 2, 2, 3, 1, 3, 1, 2, 3, 3]    # 1: Slight, 2: Serious, 3: Fatal
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# --------------------------
# Step 2: Define Variables
# --------------------------
# Features (independent variables)
X = df[['Number_of_Vehicles', 'Weather_Condition', 'Light_Condition', 'Speed_Limit']]
# Target variable (dependent variable)
y = df['Accident_Severity']

# ------------------------------
# Step 3: Split the Data
# ------------------------------
# Splitting into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------
# Step 4: Train the Linear Regression Model
# -----------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel coefficients:")
print(model.coef_)
print("Model intercept:")
print(model.intercept_)

# Evaluate the model (optional)
score = model.score(X_test, y_test)
print("\nModel R^2 score on test set:", score)

# ------------------------------------------------
# Step 5: Save the Model using joblib
# ------------------------------------------------
joblib_filename = "accident_severity_model.pkl"
joblib.dump(model, joblib_filename)
print(f"\nModel saved as '{joblib_filename}'.")

# ------------------------------------------------
# Step 6: Make a Prediction using the Saved Model
# ------------------------------------------------
# Define a hypothetical input
# For example:
#   - Number_of_Vehicles: 3
#   - Weather_Condition: 2 (Rain)
#   - Light_Condition: 1 (Daylight)
#   - Speed_Limit: 50
new_sample = pd.DataFrame({
    'Number_of_Vehicles': [3],
    'Weather_Condition': [2],
    'Light_Condition': [1],
    'Speed_Limit': [50]
})

predicted_severity = model.predict(new_sample)
print("\nPredicted Accident Severity for new sample input:")
print(predicted_severity)
