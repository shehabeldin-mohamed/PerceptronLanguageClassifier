import random
import math

class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        num_samples = len(X)
        num_features = len(X[0])

        # Initialize weights and bias randomly
        self.weights = [random.random() for _ in range(num_features)]
        self.bias = random.random()

        for _ in range(self.epochs):
            for i in range(num_samples):
                net_value = sum(X[i][j] * self.weights[j] for j in range(num_features)) - self.bias
                pred_y = self.activation(net_value)
                error = y[i] - pred_y

                # Update weights and bias
                for j in range(num_features):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias -= self.learning_rate * error

    def predict(self, X):
        net_input = sum(X[i] * self.weights[i] for i in range(len(X))) - self.bias
        return self.activation(net_input)

    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct += 1
        return (correct / len(X)) * 100.0
