from kmeans_smote import KMeansSMOTE
from TestDataGeneration.data_generation import DataGenerator
from experiment import Experiment


if __name__ == "__main__":
    total_number = 10000
    ratio = 0.9
    dg1 = DataGenerator(total_number=total_number, ratio=ratio)
    data_train, label_train = dg1.generate()
    dg2 = DataGenerator(total_number=total_number, ratio=ratio)
    data_test, label_test = dg2.generate()
    Ratio = "minority"
    kmeans_args = {"n_clusters": 20}
    imbalance_ratio_threshold = 20
    smote_args = {"k_neighbors": 10}
    kmeans_smote = KMeansSMOTE(ratio=Ratio, kmeans_args=kmeans_args, smote_args=smote_args, imbalance_ratio_threshold=imbalance_ratio_threshold)
    X_resampled, Y_resampled = kmeans_smote.fit_sample(data_train, label_train)
    exp = Experiment(data=X_resampled, label=Y_resampled)
    n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    for neighbor in n_neighbors:
        true_posi, false_posi, true_neg, false_neg = exp.get_confusion_matrix(data=data_test, label=label_test, n_neighbors=neighbor)
        print("true_posi:", true_posi,
              "false_posi:", false_posi,
              "true_neg:", true_neg,
              "false_neg:", false_neg)

