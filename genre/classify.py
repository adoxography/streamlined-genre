#!/usr/bin/env python3
import sys

import arff
import numpy as np
from sklearn.svm import LinearSVC as Classifier


GRIDS = [
    {
        'kernel': ['rbf'],
        'gamma': [1e-3, 1e-4]
    },
    {
        'kernel': ['linear'],
        'C': [1, 10, 100, 1000]
    }
]


def classify_bows(train_data, test_data):
    inputs_train, targets_train = load_data(train_data)
    inputs_test, targets_test = load_data(test_data)

    clf = init_classifier(inputs_train, targets_train)
    train_accuracy = evaluate_classifier(clf, inputs_train, targets_train)
    test_accuracy = evaluate_classifier(clf, inputs_test, targets_test)
    print('Train accuracy:', train_accuracy)
    print('Test accuracy:', test_accuracy)


def load_data(filename):
    with open(filename) as f:
        data_dict = arff.load(f)
    data = np.array(data_dict['data'])
    inputs = data[:, :-1]
    targets = data[:, -1]
    return inputs.astype(float), targets


def init_classifier(inputs, targets):
    clf = Classifier(random_state=0)
    clf.fit(inputs, targets)
    return clf


def evaluate_classifier(clf, inputs, targets):
    predictions = clf.predict(inputs)
    num_correct = sum(a == b for a, b in zip(predictions, targets))
    return num_correct / len(targets)


if __name__ == '__main__':
    _, train_data_name, test_data_name  = sys.argv
    inputs_train, targets_train = load_data(train_data_name)
    inputs_test, targets_test = load_data(test_data_name)

    clf = init_classifier(inputs_train, targets_train)
    train_accuracy = evaluate_classifier(clf, inputs_train, targets_train)
    test_accuracy = evaluate_classifier(clf, inputs_test, targets_test)
    print('Train accuracy:', train_accuracy)
    print('Test accuracy:', test_accuracy)
