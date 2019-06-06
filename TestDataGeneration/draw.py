# this is used for drawing a 2-D points picture
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


class Drawer:
    """this is a class for drawing different kind of pic

    Attributes:
        data: a numpy array
        label: a numpy array
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def plot_scatter(self):
        x = [vector[0] for vector in self.data]
        y = [vector[1] for vector in self.data]
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        color = list(map(lambda l: 'b' if l else 'g', self.label))
        ax.scatter(x=x, y=y, c=color)
        plt.show()
