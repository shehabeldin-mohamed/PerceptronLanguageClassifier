
from Perceptron import Perceptron

class SingleLayerNetwork:
    def __init__(self, classes, learning_rate, epochs):
        self.classes = classes
        self.perceptrons = [Perceptron(learning_rate, epochs) for _ in classes]

    def train(self, X, y):
        # y is expected as a list of lists with one-hot encoded labels
        for i, perceptron in enumerate(self.perceptrons):
            binary_targets = [label[i] for label in y]
            perceptron.train(X, binary_targets)

    def predict_raw(self, input_vector):
        # Return raw net values from each perceptron
        nets = []
        for perceptron in self.perceptrons:
            net_value = sum(w * x for w, x in zip(perceptron.weights, input_vector)) - perceptron.bias
            nets.append(net_value)
        return nets

    def predict(self, input_vector):
        # Return the index of the perceptron with the highest activation
        nets = self.predict_raw(input_vector)
        return nets.index(max(nets))

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

        for i, class_name in enumerate(self.classes):
            acc = 0 if total[i] == 0 else (correct[i] / total[i]) * 100.0
            print(f"Accuracy for {class_name}: {acc:.2f}%")
