# ml-from-scratch

This is a part of my assignment from my CSE 574 - Intro to ML course. This assignment is divided into four parts:

# Part 1: Data Preprocessing 

File : data_preprocess.ipynb

# Part 2: Logistic Regression using Gradient Descent 

File: log_regression_using_gd.ipynb

This project implements **Logistic Regression** (Binary Classification) from scratch in Python using NumPy, without relying on libraries like scikit-learn. It supports binary classification through gradient descent optimization and is designed to be simple and educational.

### Overview

Logistic Regression is a supervised learning algorithm used for **binary classification**. It uses the **sigmoid function** to map predicted values to probabilities and optimizes model parameters using **gradient descent**.

### Key Features

- Custom implementation using only NumPy
- Binary classification support
- Manual gradient descent
- Sigmoid activation
- Tracks loss over iterations
- Model persistence with `pickle`

### How It Works

1. **Sigmoid Function**:  
   Converts linear combinations into probabilities  
   ```
    sigmoid(z) = 1 / (1 + exp(-z))
    ```

2. **Loss Function (Binary Cross-Entropy)**:  
   Measures prediction error  
   ```
    Loss = -(1/n) * Σ [ y * log(y_pred) + (1 - y) * log(1 - y_pred) ]
    ```

3. **Gradient Descent**:  
   Updates weights to minimize the loss  
   ```
    weights = weights - learning_rate * gradient
    ```

4. **Model Training (`fit`)**:  
   - Initializes weights and bias
   - Adds bias term to the input
   - Runs gradient descent over multiple iterations
   - Logs loss every iteration

5. **Model Prediction (`predict`)**:  
   - Loads saved weights
   - Computes probability with sigmoid
   - Returns `1` if probability ≥ 0.5, else `0`

### Usage

```python
# Initialize model
model = LogitRegression(learning_rate=0.001, iterations=500000)

# Randomly initialize weights
initial_weights = np.random.rand(1 + x_train.shape[1])  # +1 for bias

# Train the model
model.fit(x_train, y_train, initial_weights)

# Predict on new data
prediction = model.predict(x_test[0])
