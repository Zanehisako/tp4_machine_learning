import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('diabetes.csv')
print(data.head())

X = data.iloc[:,0:8].to_numpy()
X = np.hstack((X,np.ones((X.shape[0],1))))
Y = data.iloc[:,8].to_numpy()
print('Features:',X.shape)
print('Features:',X)
print('Target',Y)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dervative(x):
    return sigmoid(x)*(1-sigmoid(x))

def binary_cross_entropy(y_true, y_pred):
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
  # Initialize parameters
  theta = np.random.uniform(-0.1,0.1,(X.shape[1]))# Initial weights
  theta_history = error_history = [] # To store MSE values
  for epoch in range(epochs):
      # Linear model
      z = np.dot(X,theta)
      # Predicted probabilities
      y_pred =sigmoid(z)
      # Compute gradients
      gradient = (1 / X.shape[0]) * np.dot(X.T, (y_pred- y))
      # Update weights (theta)
      theta -= gradient * learning_rate
      # Compute the cost function
      current_error = binary_cross_entropy(y,y_pred) 
      # Save current theta and error for tracking
      theta_history.append(theta)
      error_history.append(current_error)
  return theta_history, error_history, theta

_,error_history,theta=gradient_descent(X,Y)
print("error",error_history[0])
print("finished error")
plt.plot(error_history)
plt.show()
