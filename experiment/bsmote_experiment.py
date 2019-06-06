from algorithm.border_line_smote import BSMOTE
from TestDataGeneration.data_generation import DataGenerator
import numpy as np
from experiment import Experiment


if __name__ == "__main__":
    total_number = 10000
    ratio = 0.9
    k = 2
    sample_ratio = 300
    dg1 = DataGenerator(total_number=total_number, ratio=ratio)
    data_train, label_train = dg1.generate()
    dg2 = DataGenerator(total_number=total_number, ratio=ratio)
    data_test, label_test = dg2.generate()
    bsmote = BSMOTE(data=data_train, label=label_train, K=k, sample_ratio=sample_ratio)
    synthetic_data = bsmote.over_sample()
    #print(synthetic_data.shape, data_train.shape)
    data_train_with_sythetic = np.array(np.concatenate([data_train, synthetic_data], axis=0))
    label_train_with_synthetic = np.concatenate([label_train, [1.0]*len(synthetic_data)], axis=0)
    n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    for neighbor in n_neighbors:
        exp = Experiment(data=data_train_with_sythetic, label=label_train_with_synthetic)
        true_posi, false_posi, true_neg, false_neg = exp.get_confusion_matrix(data=data_test, label=label_test,
                                                                              n_neighbors=neighbor)
        print("-------------neighbor:", neighbor, "------------------")
        print("true_posi:", true_posi,
              "false_posi:", false_posi,
              "true_neg:", true_neg,
              "false_neg:", false_neg)
