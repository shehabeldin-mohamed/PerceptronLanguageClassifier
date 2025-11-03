# Perceptron Language Classifier

A simple **Perceptron-based** classifier in Python that distinguishes between four languages:
- **English**
- **Spanish**
- **German**
- **Polish**

The model uses **two perceptrons** to perform multi-class classification, extending the classic perceptron algorithm to handle multiple languages.

---

## How It Works

This project demonstrates how a **single-layer perceptron network** can be used for **language classification**.  
Each perceptron learns to recognize specific language features based on character frequency or textual patterns.

- Two perceptrons are used to separate the four languages.  
- The final classification is determined by combining their outputs.  
- Parameters such as learning rate, epochs, and dataset files are fully configurable.

---

## Installation Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/shehabeldin-mohamed/PerceptronLanguageClassifier.git
   cd PerceptronLanguageClassifier
2. **Move to src and run the program by specifying the learning rate, training dataset, and test dataset
For example:**
   ```bash
   cd src
   python Main.py 0.01 lang.train.csv lang.test.csv
