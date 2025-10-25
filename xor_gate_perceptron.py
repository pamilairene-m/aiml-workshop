import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(30)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

W1 = np.random.rand(input_neurons, hidden_neurons)
b1 = np.random.rand(1, hidden_neurons)
W2 = np.random.rand(hidden_neurons, output_neurons)
b2 = np.random.rand(1, output_neurons)

learning_rate = 0.1
epochs = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for epoch in range(epochs):
    hidden_input = np.dot(x, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += np.dot(x.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("\n--- Testing the Trained MLP for XOR ---\n")

for i in range(len(x)):
    hidden_layer = sigmoid(np.dot(x[i], W1) + b1)
    output_layer = sigmoid(np.dot(hidden_layer, W2) + b2)
    print(f"Input: {x[i]} --> Predicted: {round(float(output_layer))} (Actual: {float(output_layer):.4f})")