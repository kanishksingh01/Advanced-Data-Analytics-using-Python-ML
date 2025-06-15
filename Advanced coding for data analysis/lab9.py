#Lab 9

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Preprocess the data (if needed)
# Check for missing values
print(data.isnull().sum())

# Handle missing values if any (e.g., fill with mean or median)
# data.fillna(data.mean(), inplace=True)

# Feature Scaling
scaler = StandardScaler()
X = data.drop('Outcome', axis=1)  # Independent variables
X = scaler.fit_transform(X)
y = data['Outcome']  # Dependent variable

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model with training data (Decision Tree with max_depth=3)
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Test the model by predicting testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluating score of accuracy, precision, recall, and F1 score
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
