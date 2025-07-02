# ML from Sratch

This is a part of my assignment from my CSE 574 - Intro to ML course. This assignment is divided into four parts:

# Part 1: Data Preprocessing 

File : data_preprocess.ipynb

### Dataset Used:

1. Penguin Dataset 
2. Wine Quality Dataset
3. Emissions dataset by countries

### Data Preprocessing steps:

- Handling missing values
- Handling mismatched string formats
- Handling outliers
- Print general statistics like shape, mean, median, top percentile
- Plot graphs
- Identify and remove  unrelated features using correlation matrix
- Normalize non-categorial values

# Part 2: Logistic Regression using Gradient Descent 

File: log_regression_using_gd.ipynb
Dataset used: Penguins Dataset

This project implements **Logistic Regression** (Binary Classification) from scratch in Python using NumPy, without relying on libraries like scikit-learn. It supports binary classification through gradient descent optimization and is designed to be simple and educational.

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
```


# Part 3: Linear and Ridge Regression using OLS

File: lin_reg_ols.ipynb
Dataset used: Wine quality dataset

This project implements both **Linear Regression** and **Ridge Regression (L2 Regularization)** from scratch using **Ordinary Least Squares (OLS)** in Python.

### Linear Regression

Linear regression fits a line to minimize the **mean squared error (MSE)** between predicted and true values.

**Closed-form solution:**

$$
w = (Xᵀ X)^(-1) Xᵀ y
$$

Where:
- `X` is the input matrix (with bias column)
- `y` is the target values
- `w` is the weight vector (including bias term)

### Ridge Regression

Ridge regression adds a **regularization term** to penalize large weights and avoid overfitting.

**Closed-form solution:**

$$
w = (Xᵀ X + λI)^(-1) Xᵀ y
$$

Where:
- `λ` (lambda) controls the regularization strength
- `I` is the identity matrix (except the first entry for the bias term)

**Regularized MSE:**

$$
MSE = (1/n) * Σ (y - ŷ)^2 + 0.5 * λ * ||w||^2
$$

### Workflow

1. Load and normalize selected features from `wine_data_preprocessed.csv`
2. Train a linear regression model using closed-form OLS
3. Train a ridge regression model using regularized OLS
4. Compute Mean Squared Error (MSE) on test set
5. Find the optimal value of λ (lambda) that minimizes the MSE



# Part 4: Elastic Net Regression with Gradient Descent

File: elastic_net_reg_using_gd.ipynb
Dataset used: Emission dataset

This part implements **Elastic Net Regression** from scratch using **Gradient Descent** to model emission data for the top 20 polluted countries.

### Concept

Elastic Net combines **L1 (Lasso)** and **L2 (Ridge)** regularization:

**Loss Function**:

$$
J(w) = \frac{1}{2n} \sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda_1 \sum_{j=1}^{d} |w_j| + \frac{\lambda_2}{2} \sum_{j=1}^{d} w_j^2
$$

- Prevents overfitting and handles multicollinearity
- Promotes both sparsity and weight shrinkage

### Steps

1. **Data Preprocessing**
    - Read CSV
    - Drop global data
    - Select top 20 polluted countries
    - Normalize numeric columns
    - Map countries to numeric codes

2. **Train-Test Split**
    - 80% training, 20% testing

3. **Model Initialization**
    - Learning rate: `0.001`
    - Iterations: `200000`
    - Weight initialization: Zero / Random / Xavier

4. **Gradient Descent**
    - Implements custom weight update using MSE + L1 + L2 gradients
    - Bias term not regularized

5. **Hyperparameter Tuning**
    - Try combinations of:
        - `lambda1` ∈ [0.01, 0.05, 0.1, 0.5, 1]
        - `lambda2` ∈ [0.01, 0.05, 0.1, 0.5, 1]
        - 3 weight initialization strategies
    - Select model with **minimum final loss**

6. **Model Saving**
    - Best model weights saved using `pickle`

### Output

- Best `lambda1`, `lambda2`, and weight init strategy printed
- Trained model weights saved to `part4.p`

### Next Steps

- Evaluate model predictions on test data
- Plot loss convergence
- Extend to multi-target regression or classification if needed


## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `pickle`




