import numpy as np

# Input features (x1, x2)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target output for OR gate
y = np.array([0, 1, 1, 1])

# ----------------------------------------------
# Step 3: Initialize weights and bias randomly
# ----------------------------------------------
np.random.seed(42)            # To make results reproducible
weights = np.random.rand(2)   # Two weights for two inputs
bias = np.random.rand(1)      # One bias term
learning_rate = 0.1           # Step size for weight updates

print("Initial weights:", weights)
print("Initial bias:", bias)
print("----------------------------------")


def step_function(x):
    """Returns 1 if input >= 0, else 0"""
    return 1 if x >= 0 else 0


epochs = 10   

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    for i in range(len(X)):
    
        linear_output = np.dot(X[i], weights) + bias

    
        predicted = step_function(linear_output)

        error = y[i] - predicted

    
        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error

    
        print(f"Input: {X[i]}, Target: {y[i]}, Predicted: {predicted}")
        print(f"Updated weights: {weights}, Updated bias: {bias}")

print("\nTraining complete!")
print("----------------------------------")
print("Final weights:", weights)
print("Final bias:", bias)


print("\n--- Testing the Trained Perceptron ---")
for i in range(len(X)):
    linear_output = np.dot(X[i], weights) + bias
    predicted = step_function(linear_output)
    print(f"Input: {X[i]} --> Predicted Output: {predicted}")