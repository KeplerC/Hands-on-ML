#!/usr/bin/python

TRAIN_SET_SIZE = 60000

#fetching data
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X = mnist["data"]
y = mnist["target"]

#plotting
import matplotlib
import matplotlib.pyplot as plt

def image_of_digit(num):
    some_digit = X[num]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

class classifier:
    def __init__(self):
        self.y_train_num = 0
        self.y_train_num = 0
        
    def build_training_set(self):
        self.X_train, self.X_test, self.y_train, self.y_test = X[:TRAIN_SET_SIZE], X[TRAIN_SET_SIZE:], y[:TRAIN_SET_SIZE], y[TRAIN_SET_SIZE:]
        shuffle_index = np.random.permutation(TRAIN_SET_SIZE)
        self.X_train, self.y_train = self.X_train[shuffle_index], self.y_train[shuffle_index]
    def num_detector(self, num):
        y_train_num = (self.y_train == num)
        y_test_num = (self.y_test == num)
        self.sgd_clf = SGDClassifier(random_state=42)
        self.sgd_clf.fit(self.X_train, self.y_train_num)
    def cross_val_score(self):
        return cross_val_score(self.sgd_clf, self.X_train, self.y_train_num,cv=3, scoring="accuracy")
    
def main():
    clf = classifier()
    clf.build_training_set()
    clf.num_detector(5)
    print clf.cross_val_score()
    print confusion_matrix

if __name__ == "__main__":
    main()
