import numpy as np


class DataGenerator:
    """this class is for two dimensional data generation

    I create this class to generate 2-D data for testing
    algorithm

    Attributes:
        data: A 2-d numpy array inflated with 2-D data
        points has  both minority and majority class
        labels: A 1-D numpy array inflated with 0 or 1
        0 represent majority class, whereas 1 represent
        minority class
    """
    def __init__(self, total_number, ratio):
        """init this class instance

        :param total_number: a int, indicating how many number of points you want to generate
        :param ratio: a float number, between 0 and 1, the ratio of majority against minority
        """
        self.total_number = total_number
        self.ratio = ratio
        self.data = []
        self.labels = []

    def _generate_majority(self)->np.array:
        """this is for majority creation

        :return: a 2-D numpy array
        """
        num_of_majority = self.total_number * self.ratio
        return np.random.random((int(num_of_majority), 2)) * 100

    def _generate_minority(self)->np.array:
        """this is for minority creation

        :return: a 2-D numpy array
        """
        num_of_minority = self.total_number - self.ratio * self.total_number
        center = num_of_minority * 0.2
        left_bottom = num_of_minority * 0.25
        right_bottom = num_of_minority * 0.05
        left_top = num_of_minority * 0.2
        right_top = num_of_minority * 0.3
        center_area = 50 + (np.random.random((int(center), 2)) - 0.5) * 10
        left_bottom_area = np.array([20, 15]) - (np.random.random((int(left_bottom), 2)) - np.array([0.5, 0.5])) * 10
        right_bottom_area = np.array([90, 0]) + np.random.random((int(right_bottom), 2)) * 10
        left_top_area = (np.random.random((int(left_top), 2)) * [2, 1]) * 10 + np.array([10, 70])
        right_top_area = np.array([100, 100]) - np.random.random((int(right_top), 2)) * 15
        return np.concatenate((right_top_area, center_area, left_bottom_area, right_bottom_area, left_top_area), axis=0)

    def generate(self)->np.array:
        """generate both majority class instances and minority class instances

        :return: a 2-d numpy array
        """
        majority = self._generate_majority()
        for i in range(len(majority)):
            self.labels.append(0.)
        minority = self._generate_minority()
        for i in range(len(minority)):
            self.labels.append(1.)
        self.data, self.labels = np.concatenate((majority, minority), axis=0), np.array(self.labels)
        return self.data, self.labels

