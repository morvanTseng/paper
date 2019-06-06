from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class Over_Sample:
    def __init__(self, data, label, n, categorical_features, **kmeans_args):
        """

        :param data: 原数据
        :param label: 原数据的标签
        :param kmeans_args: 传给kmeans的参数
        :param n: 过采样率
        """
        self.data = data.tolist()
        self.label = label.tolist()
        self.kmeans = KMeans(**kmeans_args).fit(self.data, self.label)
        self.n = n
        self.categorical = categorical_features
        # 对少数类smote的近邻参数
        self.n_neighbors = 3

    def cluster_asembly(self):
        cluster_num = len(self.kmeans.cluster_centers_)
        self.point_cluster = []
        self.label_cluster = []
        for i in range(cluster_num):
            self.point_cluster.append([])
            self.label_cluster.append([])
        for index in range(len(self.kmeans.labels_)):
            cluster = self.kmeans.labels_[index]
            self.point_cluster[cluster].append(self.data[index])
            self.label_cluster[cluster].append(self.label[index])

    def examine_cluster(self, data, label):
        """

        :param data: 类簇的点
        :param label: 类簇点的标签
        :return: 0代表纯多数类簇， 2代表混合类簇， 1代表纯少数类簇
        """
        label_set = set(label)
        flag = 0
        if 0. in label_set:
            if 1. in label_set:
                flag = 2
        else:
            flag = 1
        return flag

    def synthesize(self, point_x, point_y):
        point_x = np.array(point_x)
        point_y = np.array(point_y)
        diff = point_y - point_x
        synthetic = point_x + np.random.rand() * diff
        synthetic[self.categorical] = point_x[self.categorical]
        return synthetic.tolist()

    def set(self, iterable):
        tmp = []
        for i in iterable:
            if i not in tmp:
                tmp.append(i)
        return tmp

    def synthesize_hybrid(self, data, label):
        """
        处理混合聚类簇
        :param data:
        :param label:
        :return:
        """
        major_data = []
        minor_data = []
        major_label = []
        minor_label = []
        synthetics = []
        # 将一个聚类簇的多数类和少数类分开
        for data_, label_ in zip(data, label):
            if label_ == 1.:
                minor_data.append(data_)
                minor_label.append(label_)
            else:
                major_data.append(data_)
                major_label.append(label_)
        border_minor = []
        border_major = []
        knn_major = KNeighborsClassifier(n_neighbors=1)
        knn_minor = KNeighborsClassifier(n_neighbors=1)
        knn_minor.fit(X=minor_data, y=minor_label)
        knn_major.fit(X=major_data, y=major_label)
        # 寻找边界多数类和少数类
        for major in major_data:
            index = knn_minor.kneighbors(X=[major], n_neighbors=1, return_distance=False)[0][0]
            border_minor.append(minor_data[index])
        for minor in minor_data:
            index = knn_major.kneighbors(X=[minor], n_neighbors=1, return_distance=False)[0][0]
            border_major.append(major_data[index])
        border_minor = self.set(border_minor)
        border_major = self.set(border_major)
        n_neighbors_major = self.n
        n_neighbors_minor = self.n
        if n_neighbors_minor > len(minor_data):
            n_neighbors_minor = len(minor_data)
        if n_neighbors_major > len(major_data):
            n_neighbors_major = len(major_data)
        # 合成少数类样本
        for minor in minor_data:
            if minor in border_minor:
                index = knn_major.kneighbors(X=[minor], n_neighbors=n_neighbors_major, return_distance=False)[0]
                length = len(index)
                for i in range(self.n):
                    index_ = index[i % length]
                    point_y = major_data[index_]
                    synthetics.append(self.synthesize(point_x=minor, point_y=point_y))
            else:
                index = knn_minor.kneighbors(X=[minor], n_neighbors=n_neighbors_minor, return_distance=False)[0]
                index = index[1:]
                length = len(index)
                for i in range(self.n):
                    index_ = index[i % length]
                    point_y = minor_data[index_]
                    synthetics.append(self.synthesize(point_x=minor, point_y=point_y))
        return synthetics

    def synthesize_pure(self, data, label):
        """
        处理纯少数类簇
        :param data:
        :param label:
        :return:
        """
        synthetics = []
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X=data, y=label)
        n_neighbor = self.n_neighbors
        if n_neighbor > len(data):
            n_neighbor = len(data)
        for minor in data:
            neighbors = knn.kneighbors(X=[minor], n_neighbors=n_neighbor, return_distance=False)[0]
            neighbors = neighbors[1:]
            length = len(neighbors)
            for i in range(self.n):
                index_ = neighbors[i % length]
                point_y = data[index_]
                synthetic = self.synthesize(point_x=minor, point_y=point_y)
                synthetics.append(synthetic)
        return synthetics

    def do_synthetic(self):
        self.cluster_asembly()
        synthes = []
        for cluster_point, cluster_label in zip(self.point_cluster, self.label_cluster):
            flag = self.examine_cluster(data=cluster_point, label=cluster_label)
            if flag != 0:
                synth = None
                if flag == 1:
                    synth = self.synthesize_pure(data=cluster_point, label=cluster_label)
                if flag == 2:
                    synth = self.synthesize_hybrid(data=cluster_point, label=cluster_label)
                for s in synth:
                    synthes.append(s)
        return synthes




