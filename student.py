import numpy as np

# Input features: [study_hours, attendance, assignment_score]
x = np.array([
    [1, 1, 1],   # Studies well, attends regularly, good assignments
    [1, 0, 1],   # Studies but low attendance
    [0, 1, 1],   # Poor study habits but attends well
    [0, 0, 0],   # Poor in all areas
    [1, 1, 0],   # Studies & attends but weak assignments
    [0, 1, 0],   # Attends but poor in studies & assignments
    [1, 0, 0],   # Studies only
    [0, 0, 1]    # Only good in assignments
])

# Target output (1 = Pass, 0 = Fail)
y = np.array([
    [1],  # Pass
    [1],  # Pass
    [0],  # Fail
    [0],  # Fail
    [1],  # Pass
    [0],  # Fail
    [0],  # Fail
    [0]   # Fail
])

# Initialize parameters
np.random.seed(42)
input_neurons = 3
hidden_neurons = 4
output_neurons = 1

# Initialize weights and biases
W1 = np.random.rand(input_neurons, hidden_neurons)
b1 = np.random.rand(1, hidden_neurons)
W2 = np.random.rand(hidden_neurons, output_neurons)
b2 = np.random.rand(1, output_neurons)

# Hyperparameters
learning_rate = 0.1
epochs = 10000

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training loop
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(x, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Calculate error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += np.dot(x.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("\n--- Testing the Trained Neural Network (Pass/Fail Prediction) ---\n")

# Testing
for i in range(len(x)):
    hidden_layer = sigmoid(np.dot(x[i], W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    print(f"Input (Study, Attend, Assign): {x[i]} --> Predicted: {round(float(output_layer))} (Actual value: {float(output_layer):.4f})")