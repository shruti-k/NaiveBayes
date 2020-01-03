import pandas as pd
import sys

class NaiveBayes:

    def __init__(self, train_dataset, test_dataset, model_file, result_file):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model_file = model_file
        self.result_file = result_file
        self.features = list(self.train_dataset.columns)
        self.prob_class_0 = 0
        self.prob_class_1 = 0
        self.cond_prob_model = {}
        self.prior_prob_model = {}
        self.confusion_matrix = {'0,0':0,'0,1':0,'1,0':0,'1,1':0}


    def create_model(self):
        class_0 = self.train_dataset.loc[(self.train_dataset['class'] == 0)].shape[0]
        class_1 = self.train_dataset.loc[(self.train_dataset['class'] == 1)].shape[0]
        total = (class_0 + class_1)
        # calculating individual class prob
        self.prob_class_0 = (class_0) / (total)
        self.prob_class_1 = (class_1) / (total)

        for f in self.features[:len(self.features) - 1]:
            self.model_file.write("Feature: "+str(f)+"\n")
            # finding unique values of given feature
            list = self.train_dataset[f].unique()
            feature_dict = {}
            prior_dict = {}
            # calculating conditional probabilities for class0 and class1
            # for each unique value of every feature
            for l in list:
                f0 = (self.train_dataset.loc[(self.train_dataset['class'] == 0) & (self.train_dataset[f] == l)]).shape[0]
                if f0 == 0:
                    f0 = 1
                pf0 = (f0) / (class_0)

                f1 = self.train_dataset.loc[(self.train_dataset['class'] == 1) & (self.train_dataset[f] == l)].shape[0]
                if f1 == 0:
                    f1 = 1
                pf1 = (f1) / (class_1)

                prior_prob = (self.train_dataset.loc[(self.train_dataset[f] == l)].shape[0]) / (total)

                feature_dict[str(l) + ' | 0'] = pf0
                feature_dict[str(l) + ' | 1'] = pf1
                prior_dict[str(l)] = prior_prob
                # writing prior and conditional prob for all features to model file
                self.model_file.write("P(" + str(f) + " = " + str(l) + " | Class=0) = " + str(pf0) + "\t")
                self.model_file.write("P(" + str(f) + " = " + str(l) + " | Class=1) = " + str(pf1) + "\t")
                self.model_file.write("P(" + str(f) + " = " + str(l) + ") = " + str(prior_prob))
                self.model_file.write("\n")
            self.cond_prob_model[f] = feature_dict
            self.prior_prob_model[f] = prior_dict
            self.model_file.write("---------------------------------------------\n")
        self.model_file.close()

    def perform_naive_bayes(self):
        # reading test data row wise
        for row in range(0, self.test_dataset.shape[0]):
            prob_0 = 1
            prob_1 = 1
            # calculating prob of class0 and class1 for each row
            for f in self.features[:len(self.features) - 1]:
                f0 = self.cond_prob_model[f][str(self.test_dataset.loc[row][f]) + ' | 0']
                prob_0 *= f0
                f1 = self.cond_prob_model[f][str(self.test_dataset.loc[row][f]) + ' | 1']
                prob_1 *= f1
            prob_0 *= self.prob_class_0
            prob_1 *= self.prob_class_1

            if prob_1 > prob_0:
                pred = 1
            else:
                pred = 0
            actual = self.test_dataset.loc[row]['class']
            # updating confusion matrix after each prediction
            self.confusion_matrix[str(pred) + ',' + str(actual)] = self.confusion_matrix[str(pred) + ',' + str(actual)] + 1
            # writing results to result file
            self.result_file.write(str(row+1) + "\tActual = " + str(actual) + "\tPredicted = " + str(pred) + "\n")
        self.result_file.write("Confusion Matrix\n")
        self.result_file.write("| True Negative = " + str(self.confusion_matrix['0,0']) + "| False Negative = " + str(self.confusion_matrix['0,1']) + "|\n")
        self.result_file.write("| False Positive = " + str(self.confusion_matrix['1,0']) + "| True Positive = " + str(self.confusion_matrix['1,1']) + "|\n")
        self.result_file.close()


if __name__ == "__main__":

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    mfile = sys.argv[3]
    rfile = sys.argv[4]

    train_dataset = pd.read_csv(train_file)
    test_dataset = pd.read_csv(test_file)
    model_file = open(mfile, "w")
    result_file = open(rfile, "w")

    NB = NaiveBayes(train_dataset, test_dataset, model_file, result_file)
    NB.create_model()
    NB.perform_naive_bayes()
