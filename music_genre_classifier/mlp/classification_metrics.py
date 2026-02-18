import numpy as np


class ClassificationMetrics:
    def __init__(self, y_true, y_pred, num_classes):
        self.num_classes = num_classes
        self.conf_matrix = self._confusion_matrix(y_true, y_pred)

        self._precision = None
        self._recall = None
        self._f1 = None

    def _confusion_matrix(self, y_true, y_pred):
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1

        return matrix

    def precision(self):
        if self._precision is not None:
            return self._precision

        precisions = []

        for i in range(self.num_classes):
            tp = self.conf_matrix[i, i]
            fp = self.conf_matrix[:, i].sum() - tp
            precisions.append(tp / (tp + fp) if tp + fp != 0 else 0)

        self._precision = np.array(precisions)
        return self._precision

    def recall(self):
        if self._recall is not None:
            return self._recall

        recalls = []

        for i in range(self.num_classes):
            tp = self.conf_matrix[i, i]
            fn = self.conf_matrix[i, :].sum() - tp
            recalls.append(tp / (tp + fn) if tp + fn != 0 else 0)

        self._recall = np.array(recalls)
        return self._recall

    def f1_score(self):
        if self._f1 is not None:
            return self._f1

        p = self.precision()
        r = self.recall()

        f1_scores = [
            (2 * pi * ri / (pi + ri)) if (pi + ri) != 0 else 0 for pi, ri in zip(p, r)
        ]

        self._f1 = np.array(f1_scores)
        return self._f1

    def accuracy(self):
        correct = np.trace(self.conf_matrix)
        total = self.conf_matrix.sum()
        return correct / total

    def macro_f1(self):
        return np.mean(self.f1_score())

    def weighted_f1(self):
        supports = self.conf_matrix.sum(axis=1)
        weights = supports / supports.sum()
        return np.sum(weights * self.f1_score())
