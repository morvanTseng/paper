# this is the implementation of my algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np


class My_Algorithm:
    """

    """
    def __init__(self, over_sample_ratio, kmeans_args, knn_args, categorical_features, under_sample_ratio,
                 all_minority_ratio=0.95, more_minority_less_majority_ratio=0.7, more_majority_less_minority_ratio=0.3,
                 all_majority_ratio=0.05):
        """

        :param over_sample_ratio: int, over-sample over_sample_ratio%100*numer_of_minority_class
        :param kmeans_args: dict, passed into KMeans
        :param knn_args: dict, passed to KNeighborsClassifier
        :param categorical_features: list, index of categorical_features
        :param under_sample_ratio: int, under-sample a number of under_sample_ratio % 100 of majority class
        :param all_minority_ratio: float, a threshhold under which a cluster is defined as all_minority
        :param more_minority_less_majority_ratio: float, a threshhold under which a cluster is defined as more minority
        :param more_majority_less_minority_ratio: float,  a threshhold under which a cluster is defined as less minority
        :param all_majority_ratio: float, a threshhold under which a cluster is defined as all majority
        """
        self.over_sample_ratio = over_sample_ratio
        self.kmeans_args = kmeans_args
        self.knn_args = knn_args
        self.categorical_features = categorical_features
        self.under_sample_ratio = under_sample_ratio
        self.all_minority_ratio = all_minority_ratio
        self.more_minority_less_majority_ratio = more_minority_less_majority_ratio
        self.more_majority_less_minority_ratio = more_majority_less_minority_ratio
        self.all_majority_ratio = all_majority_ratio
        self.kmeans = KMeans(**self.kmeans_args)
        self.knn = KNeighborsClassifier(**knn_args)
        self.categorical_feature_vote_neighbors = 3

    def __fit_knn_kmeans(self, data, label):
        self.data = data
        self.label = label
        self.kmeans.fit(X=data, y=label)
        self.knn.fit(X=data, y=label)
        self.number_of_minority = 0
        self.number_of_majority = 0
        for i in self.label:
            if i == 1.0:
                self.number_of_minority += 1
            else:
                self.number_of_majority += 1

    def __cluster_judge(self, cluster_point):
        """

        :param cluster_point: list, points in cluster
        :return: int, 0 for all_majority;
                     1 for more_maj_less_minor;
                     2 for balanced;
                     3 for more_minor_less_maj;
                     4 for all_minor
        """
        minority = 0
        majority = 0
        for point_index in cluster_point:
            if self.label[point_index] == 1.0:
                minority += 1
            else:
                majority += 1
        ratio = minority/(majority + minority)
        if ratio <= self.all_majority_ratio:
            return 0
        elif ratio <= self.more_majority_less_minority_ratio:
            return 1
        elif ratio <= self.more_minority_less_majority_ratio:
            return 2
        elif ratio <= self.all_minority_ratio:
            return 3
        else:
            return 4

    def __sort_clusters(self):
        self.assorted_cluster = dict({"all_minor": [],
                                      "more_minor": [],
                                      "balanced": [],
                                      "more_major": [],
                                      "all_major": []})
        length = len(self.kmeans.labels_)
        # get every cluster index
        cluster_set = set(self.kmeans.labels_)
        for cluster_index in cluster_set:
            # stockpile ever point belonging to cluster: cluster_index
            cluster_point = []
            for point_index in range(length):
                if self.kmeans.labels_[point_index] == cluster_index:
                    cluster_point.append(point_index)
            assort_result = self.__cluster_judge(cluster_point)
            if assort_result == 0:
                self.assorted_cluster["all_major"].append(cluster_point)
            elif assort_result == 1:
                self.assorted_cluster["more_major"].append(cluster_point)
            elif assort_result == 2:
                self.assorted_cluster["balanced"].append(cluster_point)
            elif assort_result == 3:
                self.assorted_cluster["more_minor"].append(cluster_point)
            else:
                self.assorted_cluster["all_minor"].append(cluster_point)

    def __get_minority_from_cluster(self, cluster_point):
        """从聚类簇中获取少数类样本

        样本点不是index，而是实实在在的点数据

        :param cluster_point: get minority samples within a given cluster
        :return: list
        """
        points = []
        for index in cluster_point:
            if self.label[index] == 1.0:
                points.append(self.data[index])
        return points

    def __calculate_density(self, cluster_point):
        """计算聚类簇的少数类样本的密度

        :param cluster_point:
        :return:
        """
        points = self.__get_minority_from_cluster(cluster_point)
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(X=points)
        sum_of_distance = kmeans.inertia_
        sparsity = sum_of_distance/len(points)
        return sparsity

    def __distribute_over_samples(self):
        """

        计算对每个含有少数类的聚类簇应该分配的采样个数
        :return:
        """
        self.distribution = dict({"all_minor": [],
                                      "more_minor": [],
                                      "balanced": [],
                                      "more_major": [],
                                      "all_major": []})
        cluster_sparsity = dict({"all_minor": [],
                                      "more_minor": [],
                                      "balanced": [],
                                      "more_major": [],
                                      "all_major": []})
        for key in self.assorted_cluster.keys():
            for cluster_point in self.assorted_cluster[key]:
                density = self.__calculate_density(cluster_point)
                cluster_sparsity[key].append(density)
        sum_of_sparsity = 0
        for key in self.assorted_cluster.keys():
            for sparsity in cluster_sparsity[key]:
                sum_of_sparsity += sparsity
        sample_number = int(self.number_of_minority * (self.over_sample_ratio/100))
        for key in cluster_sparsity.keys():
            for sparsity in cluster_sparsity[key]:
                distributed_num = round(sparsity / sum_of_sparsity * sample_number)
                self.distribution[key].append(distributed_num)

    def __categorical_feature_vote(self, point):
        """离散属性的投票值

        :param point:
        :return:
        """
        # 获取点的2k+1个近邻点
        neighbors_index = self.knn.kneighbors(X=[point], n_neighbors=self.categorical_feature_vote_neighbors, return_distance=False)[0]
        # 获取近邻点的实际坐标
        neighbors = self.data[neighbors_index]
        # 获取近邻点的离散属性值
        categorical = neighbors[:, self.categorical_features].T
        most = []
        # 计算投票
        for features in categorical:
            features_list = list(features)
            most_count = max(set(features_list), features_list.count())
            most.append(most_count)
        return most

    def __get_synthetic(self, neighbor_point, original_point):
        """获取点与点之间的合成数据

        :param neighbor_point:
        :param original_point:
        :return:

        """
        most = self.__categorical_feature_vote(original_point)
        diff = neighbor_point - original_point
        synthetic = original_point + np.random.rand() * diff
        # 更换离散型数值
        synthetic[[self.categorical_features]] = most
        return np.array(synthetic)

    def __over_sample_all_minor(self, cluster_point, over_sample_number):
        """

        :param cluster_point: index of points
        :param over_sample_number:
        :return:
        """
        synthetic = []
        number = over_sample_number
        length = len(cluster_point)
        cluster_data = np.array(self.data[cluster_point])
        cluster_label = np.array(self.data[cluster_point])
        point_index = 0
        while number:
        # 要采样的少数类样本点
            point = cluster_data[point_index]
            neighbor = self.knn.kneighbors(X=[point], n_neighbors=1, return_distance=False)[0][0]
            synthetic_point = self.__get_synthetic(neighbor_point=neighbor, original_point=point)
            synthetic.append(synthetic_point)
            number -= 1
            point_index = (point_index+1)%length
        return synthetic

    def __get_border_minority_from_more_minor_cluster(self, cluster_point_index):
        """获取多少数类聚类簇的边界多数类样本的少数类neighbor

        :param cluster_point_index:
        :return:
        """
        minor_data = []
        major_data = []
        border_minor = []
        # 将多数类样本和少数类样本分开
        for index in cluster_point_index:
            if self.label == 1.0:
                minor_data.append(self.data[index])
            else:
                major_data.append(self.data[index])
        # 获取每个多数类样本的边界少数类neighbor
        knn = KNeighborsClassifier()
        knn.fit(X=minor_data)
        for data_point in major_data:
            neighbor = knn.kneighbors(n_neighbors=1, X=[data_point])[0]
            border_minor.append(minor_data[neighbor])
        return np.array(border_minor), np.array(major_data)

    def __over_sample_more_minor_cluster(self, cluster_point_index, over_sample_number):
        """运用边界smote获取多少数聚类簇的合成点

        :param cluster_point_index:
        :param over_sample_number:
        :return:
        """
        synthetics = []
        border_minor_data, major_data = self.__get_border_minority_from_more_minor_cluster(cluster_point_index)
        border_minor_index = 0
        major_index = 0
        major_data_length = len(major_data)
        border_minor_length = len(border_minor_data)
        while over_sample_number:
            original_data = border_minor_data[border_minor_index]
            neighbor_data = major_data[major_index]
            synthetic = self.__get_synthetic(neighbor_point=neighbor_data, original_point=original_data)
            synthetics.append(synthetic)
            major_index = (1+major_index) % major_data_length
            border_minor_index = (border_minor_index+1) % border_minor_length
            over_sample_number -= 1
        return np.array(synthetics)

    def __over_sample_balanced_cluster(self, cluster_point_index, over_sample_number):
        """过采样平衡聚类簇

        :param cluster_point_index:
        :param over_sample_number:
        :return:
        """
        synthetics = []
        minor_data = []
        major_data = []
        for index in cluster_point_index:
            if self.label[index] == 1.0:
                minor_data.append(self.data[index])
            else:
                major_data.append(self.data[index])
        # 找到边界的多数类和少数类
        knn_minor = KNeighborsClassifier()
        knn_major = KNeighborsClassifier()
        knn_minor.fit(X=minor_data)
        knn_major.fit(X=major_data)
        border_minor = []
        border_maj = []
        for minor in minor_data:
            neighbor = knn_major.kneighbors(n_neighbors=1, X=[minor])[0]
            border_maj.append(minor_data[neighbor])
        for maj in border_maj:
            neighbor = knn_minor.kneighbors(X=[maj], n_neighbors=1)[0]
            border_minor.append(minor_data[neighbor])
        border_minor_length = len(border_minor)
        border_major_length = len(border_major)
        border_minor_index = 0
        border_maj_index = 0
        while over_sample_number:
            original_data = border_minor[border_minor_index]
            neighbor_data = border_maj[border_maj_index]
            synthetic = self.__get_synthetic(neighbor_point=neighbor_data, original_point=original_data)
            synthetics.append(synthetic)
            border_maj_index = (border_maj_index+1) % border_minor_length
            border_maj_index = (border_maj_index+1) % border_major_length
            over_sample_number -= 1
        return np.array(synthetics)

    def __over_sample_more_major_less_minor(self, cluster_point_index, over_sample_number):
        """合成平衡聚类簇的样本

        :param cluster_point_index:
        :param over_sample_number:
        :return:
        """
        synthetics = []
        minor = []
        major = []
        border_maj = []
        for index in cluster_point_index:
            if self.label[index] == 1.0:
                minor.append(self.data[index])
            else:
                major.append(self.data[index])
        knn_maj = KNeighborsClassifier()
        knn_maj.fit(X=major)
        for minor_data in minor:
            neighbor = knn_maj.kneighbors(X=[minor_data], n_neighbors=1, return_distance=False)[0]
            border_maj.append(major[neighbor])
        border_maj_index = 0
        minor_data_index = 0
        border_maj_length = len(border_maj)
        minor_data_length = len(minor_data)
        while over_sample_number:
            neighbor_point = border_maj[border_maj_index]
            original_point = minor[minor_data_index]
            synthetic = self.__get_synthetic(neighbor_point=neighbor_point, original_point=original_point)
            synthetics.append(synthetic)
            border_maj_index = (border_maj_index+1) % border_maj_length
            minor_data_index = (minor_data_index+1) % minor_data_length
            over_sample_number -= 1
        return np.array(synthetics)
    
    # def __get_border_point_for_more_minor_less_maj(self, cluster_point_index):
    #     """获取多少数聚类簇的边界点
    #
    #     :param cluster_index:
    #     :return:
    #     """
    #     # 多数类边界点
    #     cluster_maj_data = []
    #     # 少数类边界点
    #     cluster_minor_data = []
    #
    #     for index in cluster_point_index:
    #         if self.label[index] == 1.0:
    #             cluster_minor_data.append(self.data[index])
    #         else:
    #             cluster_maj_data.append(self.data[index])
    #     # 获取多数类的中心点
    #     kmeans = KMeans(n_clusters=1)
    #     kmeans.fit(X=cluster_maj_data)
    #     centroid = kmeans.cluster_centers_[0]
    #     # 获取距离多数类中心点最近的少数类的点
    #     for_border = cluster_minor_data.append(centroid)
    #     for_border_label = ([1.0]*len(cluster_minor_data)).append(0.0)
    #     knn = KNeighborsClassifier(n_neighbors=len(cluster_maj_data))
    #     knn.fit(X=for_border, y=for_border_label)
    #     border_minor_index = knn.kneighbors(X=[centroid], return_distance=False)[0]
    #     border_minor = for_border[border_minor_index]
    #     return np.array(border_minor), np.array(cluster_maj_data)
    #
    # def __over_sample_more_minor(self, cluster_point_index, over_sample_number):
    #     """
    #
    #     :param cluster_point_index:
    #     :param over_sample_number:
    #     :return:
    #     """
    #     synthetic = []
    #     # 获取边界点
    #     border_minor, border_maj = self.__get_border_point_for_more_minor_less_maj(cluster_point_index)
    #     minor_index = 0
    #     maj_index = 0
    #     minor_length = len(border_minor)
    #     maj_length = len(border_minor)
    #     while over_sample_number:
    #         neighbor_point = border_maj[maj_index]
    #         original_point = border_minor[minor_index]
    #         synthetic_data = self.__get_synthetic(neighbor_point=neighbor_point, original_point=original_point)
    #         synthetic.append(synthetic_data)
    #         over_sample_number -= 1
    #         minor_index = (minor_index+1)%minor_length
    #         maj_index = (maj_index+1)%minor_length
    #     return np.array(synthetic)












