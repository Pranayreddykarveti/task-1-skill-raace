#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Create a simple dataset
# Features: ['Symptom1', 'Symptom2']
# Target: ['Disease']
data = {
    'Symptom1': [0, 1, 0, 1, 0, 1, 0, 1],
    'Symptom2': [0, 0, 1, 1, 0, 0, 1, 1],
    'Disease':  [0, 0, 0, 1, 0, 1, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into features (X) and target (y)
X = df[['Symptom1', 'Symptom2']]
y = df['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['Symptom1', 'Symptom2'], class_names=['No Disease', 'Disease'], filled=True)
plt.show()


# In[ ]:




