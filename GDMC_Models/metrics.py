from copy import deepcopy
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch
import numpy as np


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def accuracy(self, predictions, labels):

        accuracy=accuracy_score(predictions, labels)
        print("accuracy is: ",accuracy)

        return accuracy

    def fmeasure(self, predictions, labels):
        f1_classwise = f1_score(predictions, labels, average=None)

        print("classwise f-1 score is",f1_classwise)
        return f1_classwise
