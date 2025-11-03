import sys
import math
import csv
from SingleLayerNetwork import SingleLayerNetwork

NUM_FEATURES = 26
LANGUAGES = ["English", "German", "Polish", "Spanish"]

def load_csv(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                lang = row[0].strip()
                text = row[1].strip()
                data.append((lang, text))
    return data

def letter_frequencies(text):
    freq = [0.0] * NUM_FEATURES
    for c in text.lower():
        if 'a' <= c <= 'z':
            freq[ord(c) - ord('a')] += 1
    return freq

def normalize(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]

def vectorize_texts(dataset):
    return [normalize(letter_frequencies(text)) for _, text in dataset]

def encode_labels(dataset):
    labels = []
    for lang, _ in dataset:
        one_hot = [1 if lang.lower() == name.lower() else 0 for name in LANGUAGES]
        labels.append(one_hot)
    return labels

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <learning_rate> <train_file> <test_file>")
        return

    try:
        learning_rate = float(sys.argv[1])
        train_file = sys.argv[2]
        test_file = sys.argv[3]

        network = SingleLayerNetwork(LANGUAGES, learning_rate, 1000)

        train_data = load_csv(train_file)
        test_data = load_csv(test_file)

        X_train = vectorize_texts(train_data)
        y_train = encode_labels(train_data)

        X_test = vectorize_texts(test_data)
        y_test = encode_labels(test_data)

        network.train(X_train, y_train)

        accuracy = network.accuracy(X_test, y_test)
        print(f"\nOverall Test Set Accuracy: {accuracy:.2f}%")

        network.print_per_class_accuracy(X_test, y_test)

        while True:
            print("\nEnter text to classify:")
            user_input = input("> ").strip()
            vector = normalize(letter_frequencies(user_input))
            prediction = network.predict(vector)
            print(f"✅ Predicted Language: {LANGUAGES[prediction]}")

            again = input("Classify another? (y/n): ").strip().lower()
            if again != "y":
                print("Exiting...")
                break

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
