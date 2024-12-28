import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("C:/Users/Administrator/Desktop/pythonprojects/E-commerce Dataset.csv")

# Fill missing values in numeric columns with the mean
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values in categorical columns with the mode (most frequent value)
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Check if there are still any missing values
if data.isna().sum().sum() > 0:
    print("There are still missing values in the data.")
else:
    print("No missing values in the data.")

# Encode categorical columns
label_encoder = LabelEncoder()

# Apply label encoding to all categorical columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Select features and target
X = data.drop(columns=['Sales'])  # Assuming 'Sales' is the target column
y = data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optional: You can print out the coefficients of the trained model
print(f'Model Coefficients: {model.coef_}')

