# this is border_line smote algorithm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class BSMOTE:
    """border_line smote

    Attributes:
        border_minority_sample: minority samples near border_line
        sample_ratio: sample sample_ratio/100*sample minority samples
        K: 2*K+1 nearest samples to determining whether a minority sample is a border-line sample
        data: all samples including majority and minority
        label: labels of each sample
    """

    def __init__(self, data, label, K, sample_ratio):
        self.data = data
        self.label = label
        self.k = 2 * K + 1
        self.sample_ratio = int(sample_ratio/100)

    def __init_knn(self):
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(self.data, self.label)

    def find_minority_samples(self):
        self.minority = []
        for i in range(len(self.label)):
            if self.label[i] == 1.0:
                self.minority.append(self.data[i])

    def find_border_line_samples(self):
        self.border_line_samples = []
        self.border_line_sample_minority = []
        all_neighbors = self.knn.kneighbors(X=self.minority, n_neighbors=self.k + 1, return_distance=False)
        minority_index = 0
        for neighbors in all_neighbors[1:]:
            flag = 0
            minority = []
            for neighbor in neighbors:
                if self.label[neighbor] == 1.0:
                    flag += 1
                    minority.append(self.data[neighbor])
            if flag > self.k/2:
                self.border_line_samples.append(self.minority[minority_index])
                self.border_line_sample_minority.append(minority)
            minority_index += 1

    def get_synthetic_data(self):
        self.synthetic_data = []
        for i in range(len(self.border_line_samples)):
            sample = self.border_line_samples[i]
            neighbors = self.border_line_sample_minority[i]
            flag = self.k
            index = 0
            length = len(neighbors)
            while flag:
                diff = sample - neighbors[index]
                synthetic = sample + np.random.rand() * diff
                self.synthetic_data.append(synthetic)
                index = (index+1) % length
                flag -= 1
        return self.synthetic_data

    def over_sample(self):
        self.__init_knn()
        self.find_minority_samples()
        self.find_border_line_samples()
        synthetic = self.get_synthetic_data()
        return np.array(synthetic)





