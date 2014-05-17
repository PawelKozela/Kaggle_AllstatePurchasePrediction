__author__ = 'Pawel Kozela'

import sys
import random
import numpy as np
import pandas as pd
import AllStatePurchaseAnalyzer
import AllstatePurchasePrediction

TEST_CASES_NB = 10


class Tester:

    headers = {}
    data_raw_train = []
    data_raw_test = []
    data_train = {}
    data_train_truncated = {}
    data_test = {}
    answers = {}

    def __init__(self):
        pass

    def load_file(self, file_name, target_list):
        with open(file_name, 'r') as data_file:
            first_line = True
            for line in data_file:
                if first_line:
                    first_line = False
                    tmp_headers = line.rstrip().split(',')
                    for i, key in enumerate(tmp_headers):
                        self.headers[key] = i
                else:
                    target_list.append(line.rstrip().split(','))

    def separate_data(self, raw_list, clean_list):
        previous_customer = ""
        customer_column = self.headers['customer_ID']
        record_type_column = self.headers['record_type']
        for line in raw_list:
            if previous_customer != line[customer_column]:
                clean_list[line[customer_column]] = []

            if line[record_type_column] == '1':
                self.answers[line[customer_column]] = line[self.headers['A']:self.headers['G']+1]
            else:
                clean_list[line[customer_column]].append(line)
                previous_customer = line[customer_column]

# ----------------------------------------------------- Truncation -----------------------------------------------------

    def evaluate_truncation_by_last_quote(self):
        correct_count = 0
        for i, key in enumerate(self.data_train_truncated):
            real_answer = self.answers[key]
            my_answer = self.data_train_truncated[key][-1][self.headers['A']:self.headers['G']+1]

            if my_answer == real_answer:
                correct_count += 1

        print correct_count / float(len(self.data_train_truncated))

    def simple_truncation_ben_s(self):
        self.data_train_truncated.clear()
        for i, key in enumerate(self.data_train):
            self.data_train_truncated[key] = []
            for j in range(0, len(self.data_train[key])):
                if j < 2:
                    self.data_train_truncated[key].append(self.data_train[key][j])
                else:
                    if random.random() < 0.3:
                        break
                    self.data_train_truncated[key].append(self.data_train[key][j])

    def test_truncation(self, truncation_method_name):
        # Truncate data_train
        truncation_method = getattr(self, truncation_method_name)
        truncation_method()

        # Evaluate
        # self.evaluate_truncation_by_last_quote()

# -------------------------------------------------------- Run  --------------------------------------------------------

    def run_test(self, test_class):
        # Load files with data
        self.load_file("train.csv", self.data_raw_train)
        self.load_file("test_v2.csv", self.data_raw_test)

        self.separate_data(self.data_raw_train, self.data_train)
        self.separate_data(self.data_raw_test, self.data_test)

        test_object = test_class()
        test_object.train(self.data_train, self.headers, self.answers)

        answers = test_object.predict(self.data_test, self.headers)

    def run_analysis(self):
        # Load files with data
        self.load_file("train.csv", self.data_raw_train)
        self.load_file("test_v2.csv", self.data_raw_test)

        self.separate_data(self.data_raw_train, self.data_train)
        self.separate_data(self.data_raw_test, self.data_test)

        for i in range(0, 1):
            self.test_truncation('simple_truncation_ben_s')

            analyzer = AllStatePurchaseAnalyzer.Analyzer(self.headers, self.data_train, self.data_train_truncated, self.data_test, self.answers)
            analyzer.run_analysis()


# -------------------------------------------------------- Main --------------------------------------------------------

def main():
    tester = Tester()
    tester.run_analysis()

    #test_class = AllstatePurchasePrediction.AllstatePurchasePredictor
    #tester.run_test(test_class)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()

