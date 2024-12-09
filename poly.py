from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = {
    'Year': [2018, 2016, 2015, 2016, 2021, 2006, 2021, 2014, 2019, 2012, 2011, 2014, 2017],
    'Speed': [159, 138, 140, 142, 179, 138, 166, 140, 151, 126, 124, 137, 138],
    'Power': [38, 15, 200, 50, 30, 80, 80, 85, 110, 150, 60, 55, 110]
}
# Create a DataFrame and Calculate the age of the cars
df = pd.DataFrame(data)
current_year = 2024
df['Age'] = current_year - df['Year']
# Define the independent variable (X) and dependent variable (y)
X = df['Age'].to_numpy().reshape(-1, 1)
Y = df['Speed'].to_numpy().reshape(-1, 1)

# Polynomial feature expansion
def generate_polynomial_features(X, degree=3):
					n_samples, n_features = X.shape
					features = [np.ones(n_samples)] # Bias term (degree 0)
					# Generate combinations of features with repetition
					for d in range(1, degree + 1):
						for combination in combinations_with_replacement(range(n_features), d):
							# Compute the product for the selected combination
							features.append(np.prod(X[:, combination], axis=1))
					return np.column_stack(features)


def Mse(y_true,y_pred):
    return np.mean((y_true - y_pred)**2)

def Mse_gradient(y_true,y_pred):
    n = len(y_true)
    return -(2/n) * np.dot(X.T ,(y_true-y_pred))

def normalize(X):
    X = np.array(X,dtype=np.float64)
    max = X.max()
    X /= max
    return X


def gradient_descent(X, y, learning_rate=0.0001, epochs=100000):
  # Initialize parameters
  theta = np.zeros((X.shape[1],1))
  print("theta",theta)
  theta_history =  [] # To store MSE values
  error_history = []
  for epoch in range(epochs):
      # Predicted probabilities
      y_pred =np.dot(X,theta)
      # Compute gradients
      gradient = Mse_gradient(y,y_pred) 
      # Update weights (theta)
      theta -= gradient * learning_rate
      # Compute the cost function
      current_error = Mse(y,y_pred) 
      # Save current theta and error for tracking
      theta_history.append(theta)
      error_history.append(current_error)
  return theta_history, error_history, theta

# Plot the results
X = normalize(X)
p =generate_polynomial_features(X,3)
yp =generate_polynomial_features(Y,3)
print(p)
print(yp)
theta_history,error_history,theta= gradient_descent(p,Y)
print("theta",theta)
plt.scatter(X, Y, color='blue', label='Data points')
predit = np.dot(p,theta)
plt.plot(X,predit , color='red', label='Polynomial fit (degree=3)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
