import numpy as np
def step_function(x):
    return 1 if x<=0 else 0

class Preceptron:
    def _init_(self,learning_rate=0.01,epochs=1000):
        self.lr=learning_rate
        self.epochs=epochs
        self.weights=None
        self.bias=None

    def train(self,X,Y):
        num_samples,num_features=X.shape
        self.weights=np.zeros(num_features)
        self.bias=0
        for _ in range(self.epochs):
            for idx,x_i in enumerate(X):
                linear_output=np.dot(x_i,self.weights)+self.bias
                y_predicted=step_function(linear_output)
                error=Y[idx]-y_predicted
                self.weights+=self.lr*error*x_i
                self.bias+=self.lr*error

    def predict(self,X):
        linear_output=np.dot(X,self.weights)
        return np.array([step_function(x) for x in linear_output])

    x=np.array([[0,0],[0,1],[1,0],[1,1]])
    y=np.array([0,0,0,1])  

    model = Preceptron(learning_rate=0.1, epochs=10)
    model.train(x, y)

    predictions =model.predict(x)
    print("Predictions:", predictions)
    print("Expected output :", y)