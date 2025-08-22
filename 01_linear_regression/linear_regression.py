"""
My First Linear Regression
Learning what happens behind sklearn.fit()
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Think of this as settings for our learning:
        - learning_rate: how big our steps are (like walking vs running)
        - n_iterations: how many times we try to improve
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.costs = []  # We'll store our mistakes here to see improvement
    
    def fit(self, X, y):
        """
        This is where learning happens!
        X = input (like years of experience)
        y = output (like salary)
        
        We're trying to find: y = mx + b
        In ML we call it: y = theta1*x + theta0
        """
        
        # How many examples do we have?
        m = len(y)  # if we have 100 people's data, m = 100
        
        # Start with random guess for our line
        # theta[0] = b (intercept)
        # theta[1] = m (slope)
        self.theta = np.zeros((2, 1))  # Start at [0, 0]
        
        # Add column of 1s to X (this is for the intercept term)
        X = np.c_[np.ones((m, 1)), X]  # Don't worry, I'll explain this
        
        # Learn: try to improve our guess n_iterations times
        for i in range(self.n_iterations):
            # Step 1: Make predictions with current guess
            # the first columns of 1s ensures that theta[0] (intercept) gets added equally to everyone:
            predictions = X.dot(self.theta)  # y = mx + b
            
            # Step 2: Calculate how wrong we were
            errors = predictions - y
            
            # Step 3: Calculate cost (average of squared errors)
            cost = np.mean(errors ** 2)
            self.costs.append(cost)
            
            # Step 4: Improve our guess
            # This is the magic - gradient descent
            gradient = (1/m) * X.T.dot(errors)
            self.theta = self.theta - self.lr * gradient
            
            # Every 100 steps, tell us how we're doing
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")



        def predict(self, X):
            """
            Use our learned line to make predictions
            """
            # Add column of 1s just like in fit
            m = len(X)
            X = np.c_[np.ones((m, 1)), X]
            return X.dot(self.theta)