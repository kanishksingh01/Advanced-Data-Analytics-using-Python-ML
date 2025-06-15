#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Individually Done - Kanishk Singh

import pandas as pd
import os

# Load the dataset
cereal_df = pd.read_csv('Cereals.csv')

# 1. Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(cereal_df.head())

# 2. Display the last 5 rows of the dataset
print("\nLast 5 rows of the dataset:")
print(cereal_df.tail())

# 3. List the dimension of the data frame (the number of rows and the number of columns)
print("\nDimensions of the data frame:")
print(cereal_df.shape)

# 4. List the variable types of each column
print("\nVariable types of each column:")
print(cereal_df.dtypes)

# 5. Display the data sorted by rating (descending)
print("\nData sorted by rating (descending):")
print(cereal_df.sort_values(by='rating', ascending=False))

# 6. Display rows where Manufacturer of cereal (mfr) is K and Calories is below 100
print("\nRows where Manufacturer of cereal (mfr) is K and Calories is below 100:")
print(cereal_df[(cereal_df['mfr'] == 'K') & (cereal_df['calories'] < 100)])

# 7. Create and display a new DataFrame from the full DataFrame but excluding fat column (without changing original DataFrame)
cereal_df_no_fat = cereal_df.drop(columns=['fat'])
print("\nDataFrame excluding fat column:")
print(cereal_df_no_fat.head())

# 8. Display rows where Potassium (potass) is missing
print("\nRows where Potassium (potass) is missing:")
print(cereal_df[cereal_df['potass'].isnull()])

# 9. Use apply() with lambda function to transform weight column that has off value (hint: columns can’t be negative)
cereal_df['weight'] = cereal_df['weight'].apply(lambda x: abs(x))
print("\nTransformed weight column to ensure no negative values:")
print(cereal_df[['weight']].head())

# 10. Strip the blank space before and after each column name, and replace the blank space in a column name with the underscore "_"
cereal_df.columns = cereal_df.columns.str.strip().str.replace(' ', '_')
print("\nColumn names after stripping spaces and replacing spaces with underscores:")
print(cereal_df.columns)

# 11. Clean categories column (mfr, type, pop_city) to display the same uppercase or lowercase
cereal_df['mfr'] = cereal_df['mfr'].str.upper()
cereal_df['type'] = cereal_df['type'].str.upper()
cereal_df['pop_city'] = cereal_df['pop_city'].str.upper()
print("\nCleaned categories columns to display uppercase:")
print(cereal_df[['mfr', 'type', 'pop_city']].head())

# 12. Use the split() method to create two new fields, one for city and one for state from pop_city column
cereal_df[['city', 'state']] = cereal_df['pop_city'].str.split(',', expand=True)
print("\nNew fields for city and state from pop_city column:")
print(cereal_df[['pop_city', 'city', 'state']].head())

# 13. Create new column that store the full name of Manufacturer of cereal (Ex. K = Kelloggs) from mfr.csv file (Hint: use merge() method)
mfr_df = pd.read_csv('mfr.csv')  # Assuming mfr.csv contains columns 'mfr' and 'mfr_name'
cereal_df = cereal_df.merge(mfr_df, on='mfr', how='left')
print("\nDataFrame with full name of Manufacturer of cereal:")
print(cereal_df[['mfr', 'mfr_name']].head())

# 14. Work with missing data in all columns, drop rows of missing data
cereal_df_cleaned = cereal_df.dropna()
print("\nDataFrame after dropping rows with missing data:")
print(cereal_df_cleaned.head())


# In[ ]:




