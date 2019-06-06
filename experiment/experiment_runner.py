from experiment import Experiment
from TestDataGeneration.data_generation import DataGenerator
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# do knn_expriment on raw data without any techniques
# metrics are f1 and accuracy
# if __name__ == "__main__":
#     total_number = 10000
#     ratio = 0.9
#     n_neighbors = [1, 3, 5, 9, 14, 20, 25, 30, 35, 40, 50]
#     cv = [3, 5, 7, 11, 13, 15, 17, 19, 21]
#     data, label = DataGenerator(total_number=total_number, ratio=ratio).generate()
#     for i in cv:
#         for j in n_neighbors:
#             exp = Experiment(data, label)
#             result = exp.do_knn_experiment(i, j)
#             print("cv:", i, " n_neighbors:", j, " acc:", np.mean(result["accuracy"]), " f1:", np.mean(result["f1"]))


if __name__ == "__main__":
    total_number = 10000
    ratio = 0.9
    n_neighbors = [1, 2, 3, 5, 7, 10, 13, 15, 17, 19]
    cv = [3, 5, 7, 9]
    data_train, label_train = DataGenerator(total_number=total_number, ratio=ratio).generate()
    data_predict, label_predict = DataGenerator(total_number=total_number, ratio=ratio).generate()
    exp = Experiment(data=data_train, label=label_train)
    for n in n_neighbors:

        true_posi, false_posi, true_neg, false_neg = exp.get_confusion_matrix(data=data_predict, label=label_predict,
                                                                              n_neighbors=n)
        print("n:", n, "true_po:", true_posi, " false_p:", false_posi, " true_neg:", true_neg, " fasle_neg:", false_neg)
