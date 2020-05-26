"""
genre.classify

Handles classification of bags of words
"""
import arff
import numpy as np
from sklearn.svm import LinearSVC as Classifier


def classify_bows(train_data, test_data):
    """
    Trains and tests a classifier on bags of words, stored in ARFF format

    :param train_data: The path to the ARFF training data file
    :param test_data: The path to the ARFF test data file
    """
    inputs_train, targets_train = load_data(train_data)
    inputs_test, targets_test = load_data(test_data)

    clf = Classifier(random_state=0)
    clf.fit(inputs_train, targets_train)

    train_accuracy = evaluate_classifier(clf, inputs_train, targets_train)
    test_accuracy = evaluate_classifier(clf, inputs_test, targets_test)

    # TODO: remove `print` side effects
    print('Train accuracy:', train_accuracy)
    print('Test accuracy:', test_accuracy)


def load_data(filename):
    """
    Loads a bag of words in ARFF format into a numpy array

    :param filename: The location of the ARFF file
    :return: A tuple of numpy.arrays, where the first element is the inputs
             and the second element is the targets
    """
    with open(filename) as bag_of_words:
        data_dict = arff.load(bag_of_words)
    data = np.array(data_dict['data'])
    inputs = data[:, :-1]
    targets = data[:, -1]
    return inputs.astype(float), targets


def evaluate_classifier(clf, inputs, targets):
    """
    Runs a classifier on `inputs` and validates them against `target`

    :param clf: The classifier to evaluate
    :param inputs: A numpy array to run through the classifier
    :param targets: The correct labels for each of the inputs
    """
    predictions = clf.predict(inputs)
    num_correct = sum(a == b for a, b in zip(predictions, targets))
    return num_correct / len(targets)
