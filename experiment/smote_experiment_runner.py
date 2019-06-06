# we generate two samples
# one for fitting the model, the other for testing
# this is a SMOTE experiment
from algorithm import  SMOTE
from TestDataGeneration.data_generation import DataGenerator
from experiment import Experiment
from algorithm.SMOTE import Smote


if __name__ == "__main__":
    total_number = 10000
    ratio = 0.9
    dg1 = DataGenerator(total_number=total_number, ratio=ratio)
    data_train, label_train = dg1.generate()
    dg2 = DataGenerator(total_number=total_number, ratio=ratio)
    data_test, label_test = dg2.generate()
    # before smote
    n_neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    for n in n_neighbors:
        print("n_neighbors:", n)
        print("before smote")
        exp = Experiment(data=data_train, label=label_train)
        true_posi, false_posi, true_neg, false_neg = exp.get_confusion_matrix(data=data_test, label=label_test, n_neighbors=n)
        print("true_posi:", true_posi,
              "false_posi:", false_posi,
              "true_neg:", true_neg,
              "false_neg:", false_neg)
        # smote
        minority_samples = []
        for i in range(len(label_train)):
            if label_train[i] == 1.0:
                minority_samples.append(data_train[i])
        smote = Smote(sample=minority_samples, N=100, k=5)
        smote.over_sampling()
        synthetics = smote.synthetic
        data_smote = list(data_train)
        data_smote.extend(synthetics)
        label_smote = list(label_train)
        label_smote.extend([1.]*len(synthetics))

        # after smote
        exp_smote = Experiment(data=data_smote, label=label_smote)
        true_posi, false_posi, true_neg, false_neg = exp_smote.get_confusion_matrix(data=data_test, label=label_test, n_neighbors=n)
        print("after smote")
        print("true_posi:", true_posi,
              "false_posi:", false_posi,
              "true_neg:", true_neg,
              "false_neg:", false_neg)


