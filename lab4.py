#!/usr/bin/env python
# coding: utf-8

# In[9]:


## Lab 4

## hello

# Prompt the user
response = input("How are you? ")

# Process the response
response = response.strip().lower()

# Check the response and print the appropriate message
if response == "ok":
    print("Good to hear!")


# In[5]:


## integer check

# Prompt the user
user_input = input("Please enter an integer: ")

# Process the input
user_input = user_input.strip()

# Check if the input is a valid integer
if user_input.isdigit():
    print("Valid integer")
else:
    print("Invalid integer")


# In[1]:


## country statistics lookup

allData = {
    'US': {'pop': 325.7, 'gdp': 19.39, 'ccy': 'USD', 'fx': 1.0},
    'CA': {'pop': 36.5, 'gdp': 1.65, 'ccy': 'CAD', 'fx': 1.35},
    'MX': {'pop': 129.2, 'gdp': 1.15, 'ccy': 'MXN', 'fx': 19.68},
}

while True:
    # Prompt the user for a country code
    country_code = input("Enter a country code (US, CA, MX) or 'exit' to quit: ").strip().upper()
    
    if country_code == 'EXIT':
        break
    
    if country_code not in allData:
        print("Invalid country code. Please try again.")
        continue
    
    # Prompt the user for a measure name
    measure_name = input("Enter a measure name (pop, gdp, ccy, fx): ").strip().lower()
    
    if measure_name not in allData[country_code]:
        print("Invalid measure name. Please try again.")
        continue
    
    # Lookup and display the value
    value = allData[country_code][measure_name]
    print(f"The {measure_name} of {country_code} is {value}.")


# In[ ]:




