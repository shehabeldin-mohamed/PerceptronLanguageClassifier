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
        self.weights = [random.random() for _ in range(num_features)]
        self.bias = random.random()

        for _ in range(self.epochs):
            for i in range(num_samples):
                net_value = sum(X[i][j] * self.weights[j] for j in range(num_features)) - self.bias
                pred_y = self.activation(net_value)
                error = y[i] - pred_y

                for j in range(num_features):
                    self.weights[j] += self.learning_rate * error * X[i][j]
                self.bias -= self.learning_rate * error

    def predict(self, X):
        net_input = sum(X[i] * self.weights[i] for i in range(len(X))) - self.bias
        return self.activation(net_input)

    def predict_raw(self, X):
        """Returns the raw weighted sum before activation."""
        return sum(X[i] * self.weights[i] for i in range(len(X))) - self.bias


class SingleLayerNetwork:
    def __init__(self, classes, learning_rate, epochs):
        self.classes = classes
        self.perceptrons = [Perceptron(learning_rate, epochs) for _ in classes]

    def train(self, X, y):
        for i, perceptron in enumerate(self.perceptrons):
            binary_targets = [row[i] for row in y]
            perceptron.train(X, binary_targets)

    def predict(self, input_vector):
        raw_outputs = [p.predict_raw(input_vector) for p in self.perceptrons]
        return raw_outputs.index(max(raw_outputs))

    def accuracy(self, X, y):
        correct = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            true_label = y[i].index(1)
            if pred == true_label:
                correct += 1
        return (correct / len(X)) * 100.0

    def print_per_class_accuracy(self, X, y):
        total = [0] * len(self.classes)
        correct = [0] * len(self.classes)

        for i in range(len(X)):
            true_label = y[i].index(1)
            total[true_label] += 1
            if self.predict(X[i]) == true_label:
                correct[true_label] += 1

        for i, cls in enumerate(self.classes):
            acc = 0 if total[i] == 0 else (correct[i] / total[i]) * 100.0
            print(f"Accuracy for {cls}: {acc:.2f}%")
