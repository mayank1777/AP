import pandas as pd
import numpy as np
import random
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def average_random_csvs(path = "./results/result_all"):
    # Generate random indices to pick from each group
    adolescent_idx = random.randint(1, 10)
    adult_idx = random.randint(1, 10)
    child_idx = random.randint(1, 10)
    
    
    # Create filenames based on the indices
    adolescent_file = f"{path}/adolescent#{adolescent_idx:03d}.csv"
    adult_file = f"{path}/adult#{adult_idx:03d}.csv"
    child_file = f"{path}/child#{child_idx:03d}.csv"
    
    # Read the CSVs into DataFrames
    adolescent_df = pd.read_csv(adolescent_file)
    adult_df = pd.read_csv(adult_file)
    child_df = pd.read_csv(child_file)
    
    # Ensure that the DataFrames have the same shape
    if not (adolescent_df.shape == adult_df.shape == child_df.shape):
        raise ValueError("The CSV files do not have the same shape. Ensure all CSVs have identical columns and rows.")
    adolescent_df.drop(adolescent_df.tail(1).index,inplace=True)
    adult_df.drop(adult_df.tail(1).index,inplace=True)
    child_df.drop(child_df.tail(1).index,inplace=True)
    adolescent_df = adolescent_df.drop("Time", axis = 1)
    adult_df = adult_df.drop("Time", axis = 1)
    child_df = child_df.drop("Time", axis = 1)

    # Compute the average of the DataFrames
    average_df = (adolescent_df + adult_df + child_df) / 3
    # average_df = 
    
    return average_df

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    
    # Shuffle indices
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))
    
    # Split the data
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

def train_regressor(df):
    # Define features and target
    X = df[['BG', 'CGM', 'CHO']]
    y = df['insulin']
    # print(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

    # Create polynomial features
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit a linear regression model to the polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predict using the test set
    y_pred = model.predict(X_test_poly)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Optional: Print coefficients and intercept
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    
class PIDRegressor:
    def __init__(self, Kp=0.1, Ki=0.01, Kd=0.05, learning_rate=0.001):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.learning_rate = learning_rate
        self.integral = 0
        self.prev_error = None
    
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            total_error = 0
            self.integral = 0
            self.prev_error = None
            self.target = 100
            
            for i in range(len(X)):
                # Calculate error
                error = X.iloc[i][0] - self.target
                total_error += error ** 2
                
                # Proportional term
                P = self.Kp * error
                
                # Integral term
                self.integral += error # sample time = 1 min
                I = self.Ki * self.integral
                
                # Derivative term
                D = 0 if self.prev_error is None else self.Kd * (error - self.prev_error)
                
                # PID output
                output = P + I + D
                
                # Update PID parameters
                self.Kp += self.learning_rate * error * P
                self.Ki += self.learning_rate * error * I
                self.Kd += self.learning_rate * error * D
                
                # Update previous error
                self.prev_error = error
            
            mse = total_error / len(X)
            print(f"Epoch {epoch+1}/{epochs}, Mean Squared Error: {mse}")
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            # Calculate error (assume error to be zero for prediction purpose)
            error = 0
            
            # Proportional term
            P = self.Kp * error
            
            # Integral term
            I = self.Ki * self.integral
            
            # Derivative term
            D = 0 if self.prev_error is None else self.Kd * (error - self.prev_error)
            
            # PID output
            output = P + I + D
            predictions.append(output)
        
        return np.array(predictions)

# Initialize PID controller with hyperparameters
# pid_model = PIDRegressor(Kp=0.1, Ki=0.01, Kd=0.05, learning_rate=0.001)

# # Fit the PID controller
# pid_model.fit(X_train, y_train, epochs=50)

# # Predict using the test set
# y_pred = pid_model.predict(X_test)

# # Calculate the mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Custom train_test_split function
def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    
    # Shuffle indices
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))
    
    # Split the data
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

# Load the dataset
df = pd.read_csv('path/to/your/dataset.csv')

# Define features and target
X = df[['BG', 'CGM', 'CHO']]
y = df['insulin']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)

# Define Polynomial Regression Model
class PolynomialRegressionModel(nn.Module):
    def __init__(self, degree=2):
        super(PolynomialRegressionModel, self).__init__()
        self.poly = PolynomialFeatures(degree)
        self.linear = nn.Linear(self.poly.fit_transform(X_train).shape[1], 3)  # Output Kp, Ki, Kd
    
    def forward(self, x):
        poly_x = self.poly.fit_transform(x)
        poly_x = torch.tensor(poly_x, dtype=torch.float32)
        return self.linear(poly_x)

# PID Controller Function
class PIDController:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = None
    
    def compute(self, kp, ki, kd, x):
        error = 100 - x[0]  # Using first feature (BG) and target constant (100)

        # Proportional term
        P = kp * error

        # Integral term (accumulates over time)
        self.integral += error
        I = ki * self.integral

        # Derivative term (based on previous error)
        D = 0 if self.prev_error is None else kd * (error - self.prev_error)

        # Update previous error
        self.prev_error = error

        return P + I + D

# Model, Loss, and Optimizer Setup
degree = 2
model = PolynomialRegressionModel(degree)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training Loop
pid = PIDController()
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    kp_ki_kd = model(X_train_tensor)
    insulin_preds = []

    # Using gradients during PID computation
    for i in range(len(X_train_tensor)):
        kp, ki, kd = kp_ki_kd[i]
        insulin_pred = pid.compute(kp, ki, kd, X_train_tensor[i])
        insulin_preds.append(insulin_pred)
    
    insulin_preds = torch.stack(insulin_preds)
    
    # Compute loss
    loss = criterion(insulin_preds, y_train_tensor)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Print loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    kp_ki_kd = model(torch.tensor(X_test.values, dtype=torch.float32))
    insulin_preds = []
    for i in range(len(X_test)):
        kp, ki, kd = kp_ki_kd[i]
        insulin_pred = pid.compute(kp, ki, kd, torch.tensor(X_test.iloc[i].values, dtype=torch.float32))
        insulin_preds.append(insulin_pred)
    
    mse = mean_squared_error(y_test, insulin_preds)
    print(f"Mean Squared Error on Test Set: {mse}")






if __name__ == "__main__":
    epochs = 10
    for _ in tqdm(range(epochs)):
        path = "./results/result_all"
        average_df = average_random_csvs(path)
        # print(average_df)
        train_regressor(average_df)
