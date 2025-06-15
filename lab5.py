#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the data from contacts.csv and set email as index
df = pd.read_csv("contacts.csv", index_col="email")

# Display the data
print("a. Dataframe:")
print(df.to_string())  # Improved formatting for better readability

# Get information about the dataframe
print("\nb. Information:")
df.info()

# Get number of rows and columns
print("\nc. Number of rows and columns:")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Using shape: {df.shape}")  # Alternative using shape

# Display first and last rows
print("\nd. First and last two rows:")
print(df.head(2))
print(df.tail(2))

# Access specific columns and rows
print("\ne. Phone column:")
print(df["phone"])

print("\nf. First and Last name columns:")
print(df[["first", "last"]])  # Efficient selection using list

# Access specific cell by email
email_to_find = "smsmith@yahoo.com"
if email_to_find in df.index:
    print(f"\ng. Phone number for email={email_to_find}: {df.loc[email_to_find, 'phone']}")
else:
    print(f"Email '{email_to_find}' not found in the dataframe.")


# In[ ]:




