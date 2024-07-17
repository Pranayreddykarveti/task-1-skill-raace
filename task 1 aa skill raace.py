#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize dataset
# Example: XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Set seed for reproducibility
np.random.seed(42)

# Network parameters
input_layer_neurons = X.shape[1]
hidden_layer_neurons = 2
output_layer_neurons = y.shape[1]
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wo = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bo = np.random.uniform(size=(1, output_layer_neurons))

# Training the neural network
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_activation = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_activation, wo) + bo
    predicted_output = sigmoid(output_layer_input)
    
    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(wo.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
    
    # Updating Weights and Biases
    wo += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    bo += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    if epoch % 1000 == 0:
        loss = np.mean(np.square(y - predicted_output))
        print(f'Epoch {epoch} Loss: {loss}')

# Testing the trained model
print("Predicted Output:\n", predicted_output)


# In[ ]:




