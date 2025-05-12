import numpy as np
import pandas as pd

''' to implement a neural network from scratch i will need following arch:
IP -> hidden layer -> output layer

IP = take numpy array of shape (1,m)
hL1= got weights w= [w1,w2,w3.....] and b = [b1,b2,b3....]
hl1 will find z=w.x+b or z= np.fot(w,x)+b
hl1 will apply activation let say umm sigmoid a= 1/(1+e^-z) or relu (y=x for x>0 else 0)
output will be a= [a1,a2,a3....]

now give it as input to h2
hL2= got weights w= [w1,w2,w3.....] and b = [b1,b2,b3....]
hl2 will find z=w.x+b or z= matmul(w,x)+b
hl2 will apply activation let say umm sigmoid a= 1/(1+e^-z)
output will be a= [a1,a2,a3....]

this will go to output layer sigmoid k=1/(1+e^-a)

then backprop will be applied by chain rule
'''

class NeuralNetwork:
    def __init__(self, input_mat, w1, b1, w2, b2, hidden, output):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.input_mat = input_mat
        self.hidden = hidden
        self.output = output

    def forward(self, input_mat):
        self.z1 = np.dot(input_mat, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, y_true, y_pred):
        self.dz2 = y_pred - y_true
        self.dw2 = np.dot(self.a1.T, self.dz2) / y_true.shape[0]
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True) / y_true.shape[0]
        self.dz1 = np.dot(self.dz2, self.w2.T) * self.relu_derivative(self.z1)
        self.dw1 = np.dot(self.input_mat.T, self.dz1) / y_true.shape[0]
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True) / y_true.shape[0]
        return self.dw1, self.db1, self.dw2, self.db2
    
    def update_weights(self, learning_rate):
        self.w1 -= learning_rate * self.dw1
        self.b1 -= learning_rate * self.db1
        self.w2 -= learning_rate * self.dw2
        self.b2 -= learning_rate * self.db2
        return self.w1, self.b1, self.w2, self.b2

    def relu(self, x):
        return np.maximum(0, x)
        
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
        accuracy = correct_predictions / y_true.shape[0]
        return accuracy


class Model:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w1 = np.random.rand(input_dim, hidden_dim) * 0.1
        self.b1 = np.random.rand(1, hidden_dim)
        self.w2 = np.random.rand(hidden_dim, output_dim) * 0.1
        self.b2 = np.random.rand(1, output_dim)
    
    def train(self, input_mat, y_true, nn, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = nn.forward(input_mat)
            accuracy = nn.accuracy(y_true, y_pred)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}")
            
            nn.backward(y_true, y_pred)
            nn.update_weights(learning_rate)
        return nn.w1, nn.b1, nn.w2, nn.b2

    def predict(self, input_mat, w1, b1, w2, b2):
        nn = NeuralNetwork(input_mat, w1, b1, w2, b2, self.hidden_dim, self.output_dim)
        y_pred = nn.forward(input_mat)
        return y_pred


if __name__ == "__main__":
    input_dim = 300
    hidden_dim = 100
    output_dim = 10
    epochs = 100
    learning_rate = 0.01

    input_mat = np.random.rand(1, input_dim)
    y_true = np.zeros((1, output_dim))
    y_true[0, np.random.randint(0, output_dim)] = 1

    model_instance = Model(input_dim, hidden_dim, output_dim)
    w1, b1, w2, b2 = model_instance.w1, model_instance.b1, model_instance.w2, model_instance.b2
    nn = NeuralNetwork(input_mat, w1, b1, w2, b2, hidden_dim, output_dim)

    w1, b1, w2, b2 = model_instance.train(input_mat, y_true, nn, epochs, learning_rate)
    y_pred = model_instance.predict(input_mat, w1, b1, w2, b2)
    print("Predicted output:", y_pred)
    print("Predicted class:", np.argmax(y_pred))

