#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import matplotlib.pyplot as plt

# 1) Read data from sales.csv into a DataFrame
df = pd.read_csv('sales.csv')

# 2) Review the data using head()
print(df.head())

# 3) Summarize sales by product using a pivot table
pivot_product = pd.pivot_table(df, values='count', index='product', aggfunc='sum')
print(pivot_product)

# 4) Summarize sales by product and region using a pivot table
pivot_product_region = pd.pivot_table(df, values='count', index='product', columns='region', aggfunc='sum')
print(pivot_product_region)

# 5) From the two-dimensional pivot table, create a stacked horizontal bar chart
pivot_product_region.plot(kind='barh', stacked=True)
plt.xlabel('Count of Products Purchased')
plt.ylabel('Product')
plt.title('Sales by Product and Region')
plt.show()


# In[ ]:




