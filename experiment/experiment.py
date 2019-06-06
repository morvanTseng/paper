# this is a first-step test
# i use my sythetic 2-D data which is imbalanced to test
# how the k-means algorithm performs without any oversampling
# or undersampling techniques
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class Experiment:
    """any experiment can use this class

    Attributes:
        data: input data for experiment
        label: input label for experiment
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def do_knn_experiment(self, cv, n_neighbors)->dict:
        scorings = ["accuracy", "f1"]
        result = {}
        for scoring in scorings:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            result[scoring] = cross_val_score(estimator=knn, X=self.data, y=self.label, scoring=scoring, cv=cv)
        return result

    def get_confusion_matrix(self, data, label, n_neighbors)->list:
        """get confusion matrix

        we train an model by using inner self.data and self.label
        and we get confusion matrix on trained model from data and label

        :param data: np.array
        :param label: np.array
        :return: confusion matrix
        """

        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(self.data, self.label)
        true_posi = 0
        true_neg = 0
        false_posi = 0
        false_neg = 0
        prediction = model.predict(data)
        for i, j in zip(prediction, label):
            if i == 0:
                if j == 0:
                    true_neg += 1
                else:
                    false_neg += 1
            else:
                if j == 0:
                    false_posi += 1
                else:
                    true_posi += 1

        return true_posi, false_posi, true_neg, false_neg

