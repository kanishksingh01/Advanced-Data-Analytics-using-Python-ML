#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# File path for the uploaded file
file_path = r"C:\Users\kanis\Documents\Univ\advanced-coding\group project\cleaned_data.csv"


# Step 1: Load the data
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the path.")
    exit()

# Display initial data information
print("Initial Data:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Remove missing values if any
data = data.dropna()

# Print data info and descriptive statistics
print("\nData After Cleaning:")
print(data.info())

print("\nDescriptive Statistics:")
print(data.describe())

# Display column names
print("\nColumn Names:")
print(data.columns)

# Columns to be used in regression and decision tree
# These can be updated based on the actual column names in the uploaded file
# Assuming the first column is the feature (X) and the second column is the target (y)
columns = data.columns.tolist()
column1 = columns[0]  # First column as feature
column2 = columns[1]  # Second column as target

# Check if the columns exist in the dataframe
if column1 in data.columns and column2 in data.columns:
    # Check if both columns are numeric
    if pd.api.types.is_numeric_dtype(data[column1]) and pd.api.types.is_numeric_dtype(data[column2]):
        
        # Plot histogram of column1
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column1], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {column1}')
        plt.xlabel(column1)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
        # Plot scatter plot of column1 vs column2
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=column1, y=column2, color='green')
        plt.title(f'Scatter Plot of {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.tight_layout()
        plt.show()
        
        # Step 1: Train-test split for regression and decision tree
        X = data[[column1]]
        y = data[column2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 2: Perform linear regression using sklearn
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = linear_model.predict(X_test)

        # Print regression metrics
        print("\nLinear Regression Metrics:")
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")

        print("\nRegression Coefficients:")
        print(f"Intercept: {linear_model.intercept_}")
        print(f"Slope: {linear_model.coef_[0]}")

        # Plot regression line
        sns.regplot(x=column1, y=column2, data=data, line_kws={'color': 'red'})
        plt.title(f'Regression Plot of {column1} vs {column2}')
        plt.xlabel(column1)
        plt.ylabel(column2)
        plt.tight_layout()
        plt.show()

        # Step 3: Perform regression using statsmodels for a detailed summary
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        
        print("\nRegression Summary:")
        print(model_sm.summary())

        # Step 4: Decision Tree Regressor
        tree_model = DecisionTreeRegressor(random_state=42, max_depth=5)  # Max depth to avoid overfitting
        tree_model.fit(X_train, y_train)

        # Predict on test data
        y_tree_pred = tree_model.predict(X_test)

        # Print decision tree metrics
        print("\nDecision Tree Regression Metrics:")
        print(f"R² Score: {r2_score(y_test, y_tree_pred):.4f}")
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_tree_pred):.4f}")
        print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_tree_pred):.4f}")

        # Visualize the decision tree structure
        plt.figure(figsize=(20, 10))
        plot_tree(tree_model, feature_names=[column1], filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree Regressor")
        plt.show()

    else:
        print(f"Columns {column1} and {column2} must be numeric for regression and decision tree modeling.")
else:
    print(f"Columns {column1} and {column2} not found in the dataset.")


# In[ ]:




