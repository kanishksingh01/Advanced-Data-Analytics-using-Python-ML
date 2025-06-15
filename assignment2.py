#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
import string
import matplotlib.pyplot as plt
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# -----------------------------------------------
# Predictive Modeling: BostonHousing.csv
# -----------------------------------------------

# 1. Read the BostonHousing.csv dataset
boston_data = pd.read_csv("BostonHousing.csv")

# 2. Summary statistics for each column
print("Summary Statistics:")
print(boston_data.describe())

# 3. Create dependent and independent arrays
X = boston_data.drop(columns=['MEDV'])
y = boston_data['MEDV']

# 4. Split data into training (60%) and validation (40%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# 5. Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Assess model performance
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"Training MSE: {mse_train}, Validation MSE: {mse_val}")
print(f"Training R2: {r2_train}, Validation R2: {r2_val}")

# -----------------------------------------------
# Text Analysis: hotel-reviews.csv
# -----------------------------------------------

# 7. Read the hotel-reviews.csv dataset
hotel_data = pd.read_csv("hotel-reviews.csv")

# 8. Combine all reviews into one string
text = " ".join(hotel_data['text'])

# 9. Tokenize the text
tokens = word_tokenize(text)

# 10. Refine the text: Remove non-letters, stopwords, and convert to lowercase
stop_words = set(stopwords.words('english'))
refined_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

# 11. Count the number of occurrences of each word
freq_dist = FreqDist(refined_tokens)
print("Most Common Words:", freq_dist.most_common(30))

# 12. Display a word frequency line chart (first 30 words)
freq_dist.plot(30, title="Top 30 Word Frequencies")

# 13. Sentiment analysis using VADER on the first review
sia = SentimentIntensityAnalyzer()
first_review_sentiment = sia.polarity_scores(hotel_data['text'][0])
print("Sentiment Analysis of First Review:", first_review_sentiment)

# 14. Define a function to calculate compound score
def compound_score(text):
    return sia.polarity_scores(text)['compound']

# 15. Create a new column for compound scores
hotel_data['compound'] = hotel_data['text'].apply(compound_score)

# Display the updated dataframe
print(hotel_data[['text', 'compound']].head())

# Save the updated DataFrame as a new CSV file
hotel_data.to_csv("hotel_reviews_with_compound.csv", index=False)
print("Updated hotel reviews saved to 'hotel_reviews_with_compound.csv'.")


# In[ ]:




