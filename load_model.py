import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('KNNmodel.pkl')  # Ensure this is the model, not data

# Load the test data
test_data = pd.read_csv('Crop_recommendationlabel.csv')

# Assuming the last column is the target
X_test = test_data.iloc[:, 0:7]
y_test = test_data.iloc[:, 8]

# Generate test predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy to a file
with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}')

print(f'Accuracy: {accuracy}')