from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from copy import deepcopy


class Under_Sample:

    """
    将生成的样本点与原多数类样本合在一起，使用kmeans.n_cluster参数和过采样的时候一样
    将所有的聚类簇分为三类
    纯多数类：进行欠采样，欠采样率人为设置
    混合类：不做给改变
    纯少数类不做改变
    """
    def __init__(self, major, major_label, synthetics, synthetics_label, categorical_features, rate, **kmeans_args):
        self.major = major
        self.major_label = major_label
        self.synthetics = synthetics
        self.synthetics_label = synthetics_label
        self.kmeans_args = kmeans_args
        self.categorical = categorical_features
        self.undersample_rate = rate

    def concate_major_and_minor(self):
        """
        将多数类和少数类合并起来
        :return:
        """
        self.major_synthetics = self.major + self.synthetics
        self.major_synthetics_label = self.major_label + self.synthetics_label

    def do_kmeans(self):
        """
        将数据集用kmeans方法聚类， 类簇个数和合成时候一样
        :return:
        """
        self.kmeans = KMeans(**self.kmeans_args)
        self.kmeans.fit(X=self.major_synthetics, y=self.major_synthetics_label)

    def hybrid_precessor(self, data, label):
        major = []
        major_label = []
        minor = []
        minor_label = []
        border_major = []
        border_major_label = []
        for data_, label_ in zip(data, label):
            if label == 1.0:
                minor.append(data_)
                minor_label.append(label_)
            else:
                major.append(data_)
                major_label.append(label_)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X=major, y=major_label)
        for m in minor:
            neighbor = knn.kneighbors(X=[m], n_neighbors=1, return_distance=False)[0][0]
            border_major.append(major[neighbor])
            border_major_label.append(0.0)
        return_data = []
        return_label = []
        for d in major:
            if d not in border_major:
                return_data.append(d)
                return_label.append(0.0)
        return return_data, return_label, border_major, border_major_label

    def prepare_clusters(self):
        """
        将所有需要欠采样的聚类簇做好处理
        :return:
        """
        total_cluster = len(self.kmeans.cluster_centers_)
        data_clusters = [list() for i in range(total_cluster)]
        label_clusters = [list() for i in range(total_cluster)]
        # 将聚类后的点分到自己的聚类簇区
        for index in range(len(self.kmeans.labels_)):
            cluster_index = self.kmeans.labels_[index]
            point = self.major_synthetics[index]
            label = self.major_synthetics_label[index]
            data_clusters[cluster_index].append(point)
            label_clusters[cluster_index].append(label)

        # 处理聚类簇
        self.prepared_clusters = []
        self.prepared_labels = []
        self.borders = []
        self.borders_label = []
        for datas, labels in zip(data_clusters, label_clusters):
            labels_set = set(labels)
            if len(labels_set) == 1:
                if 0. in labels_set:
                    self.prepared_clusters.append(datas)
                    self.prepared_labels.append(labels)
            else:
                d, l, bd, bl = self.hybrid_precessor(data=datas, label=labels)
                self.prepared_clusters.append(d)
                self.prepared_labels.append(l)
                self.borders.append(bd)
                self.borders_label.append(bl)

    def points_merge(self, point_x, point_y):
        """
        用来将两个点合成
        :param point_x:
        :param point_y:
        :return:
        """
        x = np.array(point_x)
        y = np.array(point_y)
        x_ = (x + y) / 2
        x_[[self.categorical]] = x[[self.categorical]]
        return x_.tolist()

    def undersample_clusters(self, cluster):
        cluster_length = len(cluster)
        copy = deepcopy(cluster)
        under_sample = int(cluster_length * self.undersample_rate)
        if under_sample < 1:
            under_sample = 1
        flag = cluster_length
        while flag > under_sample:
            x = copy[0]
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X=copy, y=[0.0] * flag)
            y_index = knn.kneighbors(X=[x], n_neighbors=2, return_distance=False)[0][1]
            y = copy[y_index]
            z = self.points_merge(x, y)
            copy.append(z)
            copy.remove(x)
            copy.remove(y)
            flag = flag - 1
        return copy

    def do_undersample(self):
        # 将传入的原多数类样本点和smote合成的少数类样本点合并成新的数据集
        self.concate_major_and_minor()
        # 将合并成的新数据集作用与kmeans算法生成与原来相同个数的类簇
        self.do_kmeans()
        # 处理好每个聚类簇
        # 对于纯多数类簇采用欠采样方法，混合类簇采用找出边界样本点，纯少数类簇不进行操作
        self.prepare_clusters()
        # 返回的最终所有多数类
        major_samples = []
        # 欠采样每一个准备好的类簇
        for cluster in self.prepared_clusters:
            # 获取欠采样结果
            result = self.undersample_clusters(cluster)
            for s in result:
                major_samples.append(s)
        # 将所有的边界多数类放入返回结果中
        for cluster in self.borders:
            for point in cluster:
                major_samples.append(point)
        return major_samples
