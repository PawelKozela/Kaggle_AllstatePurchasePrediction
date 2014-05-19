__author__ = 'Pawel'

import math
import csv
import random
import numpy as np
import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

CAR_VALUES = {
    '': -1,
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
}

STATES = {
    '': -1,
    'AL': 0,
    'AR': 1,
    'CO': 2,
    'CT': 3,
    'DC': 4,
    'DE': 5,
    'FL': 6,
    'GA': 7,
    'IA': 8,
    'ID': 9,
    'IN': 10,
    'KS': 11,
    'KY': 12,
    'MD': 13,
    'ME': 14,
    'MO': 15,
    'MS': 16,
    'MT': 17,
    'ND': 18,
    'NE': 19,
    'NH': 20,
    'NM': 21,
    'NV': 22,
    'NY': 23,
    'OH': 24,
    'OK': 25,
    'OR': 26,
    'PA': 27,
    'RI': 28,
    'SD': 29,
    'TN': 30,
    'UT': 31,
    'WA': 32,
    'WI': 33,
    'WV': 34,
    'WY': 35,
}


class Analyzer():

    logger = None

    headers = {}
    data_train = {}
    data_train_truncated = {}
    data_train_truncated_train = {}
    data_train_truncated_test = {}
    data_test = {}
    answers = {}

    location_customers_count = {}
    location_average_A = {}
    location_average_B = {}
    location_average_C = {}
    location_average_D = {}
    location_average_E = {}
    location_average_F = {}
    location_average_G = {}
    location_average_C_previous = {}
    location_average_price = {}
    location_G_count = {}

    NB_FEATURES = 7

    NB_CV_SUBSETS = 5

    ANALYSIS = True

    # Tuning parameters
    MAX_DEPTH = 5000
    NB_TREES = 100
    MIN_SAMPLES_LEAF = 10
    MAX_FEATURES = 7
    NB_JOBS = 3
    NB_RUNS = 1
    CRITERION = 'gini'
    TESTING_OPTION = 6

    def __init__(self, headers, data_train, data_train_truncated, data_test, answers):
        self.headers = headers
        self.data_test = data_test
        self.data_train = data_train
        self.data_train_truncated = data_train_truncated
        self.answers = answers

    # -------------------------------------------------------- Random Forests --------------------------------------------------------

    def prepare_location_stats(self, to_train_ids):
        self.location_customers_count.clear()
        self.location_average_A.clear()
        self.location_average_B.clear()
        self.location_average_C.clear()
        self.location_average_D.clear()
        self.location_average_E.clear()
        self.location_average_F.clear()
        self.location_average_G.clear()
        self.location_average_C_previous.clear()
        self.location_average_price.clear()

        self.location_G_count.clear()

        can_use = {}

        for i, key in enumerate(self.data_train_truncated):
            loc = self.data_train_truncated[key][0][self.headers['location']]
            if loc not in self.location_customers_count:
                self.location_customers_count[loc] = 0
                self.location_average_A[loc] = 0
                self.location_average_B[loc] = 0
                self.location_average_C[loc] = 0
                self.location_average_D[loc] = 0
                self.location_average_E[loc] = 0
                self.location_average_F[loc] = 0
                self.location_average_G[loc] = 0
                self.location_average_C_previous[loc] = 0
                self.location_average_price[loc] = 0
                self.location_G_count[loc] = np.zeros(shape=(5,))

            self.location_customers_count[loc] += 1
            self.location_average_A[loc] += float(self.answers[key][0])
            self.location_average_B[loc] += float(self.answers[key][1])
            self.location_average_C[loc] += float(self.answers[key][2])
            self.location_average_D[loc] += float(self.answers[key][3])
            self.location_average_E[loc] += float(self.answers[key][4])
            self.location_average_F[loc] += float(self.answers[key][5])
            self.location_average_G[loc] += float(self.answers[key][6])
            self.location_average_price[loc] += float(self.data_train[key][-1][self.headers['cost']])
            if self.data_train_truncated[key][-1][self.headers['C_previous']] == 'NA':
                self.data_train_truncated[key][-1][self.headers['C_previous']] = 0
            self.location_average_C_previous[loc] += float(self.data_train_truncated[key][-1][self.headers['C_previous']])
            self.location_G_count[loc][int(self.answers[key][6])] += 1

    def evaluate_single_option_split(self, ids, rf_answers, last_quote_answers, real_answers):
        answers_matrix = np.zeros(shape=(7, 5, 5, 5))

        for j in range(0, len(rf_answers)):
            for i in range(0, 7):
                answers_matrix[i][int(rf_answers[j, i])][int(last_quote_answers[j, i])][int(real_answers[j, i])] += 1

        common_good = np.zeros(shape=(7,))
        common_errors = np.zeros(shape=(7,))
        uncommon_errors = np.zeros(shape=(7,))
        rf_errors = np.zeros(shape=(7,))
        last_quote_errors = np.zeros(shape=(7,))

        for option_id in range(0, 7):
            for rf_answer_id in range(0, 5):
                for last_quote_answer_id in range(0, 5):
                    for real_answer_id in range(0, 5):
                        if rf_answer_id == last_quote_answer_id and rf_answer_id == real_answer_id:
                            common_good[option_id] += answers_matrix[option_id][rf_answer_id][last_quote_answer_id][real_answer_id]

                        if rf_answer_id == last_quote_answer_id and last_quote_answer_id != real_answer_id:
                            common_errors[option_id] += answers_matrix[option_id][rf_answer_id][last_quote_answer_id][real_answer_id]

                        if rf_answer_id != last_quote_answer_id and rf_answer_id != real_answer_id and last_quote_answer_id != real_answer_id:
                            uncommon_errors[option_id] += answers_matrix[option_id][rf_answer_id][last_quote_answer_id][real_answer_id]

                        if rf_answer_id != last_quote_answer_id and rf_answer_id == real_answer_id:
                            last_quote_errors[option_id] += answers_matrix[option_id][rf_answer_id][last_quote_answer_id][real_answer_id]

                        if rf_answer_id != last_quote_answer_id and last_quote_answer_id == real_answer_id:
                            rf_errors[option_id] += answers_matrix[option_id][rf_answer_id][last_quote_answer_id][real_answer_id]

        for option_id in range(0, 7):
            total = common_good[option_id] + common_errors[option_id] + uncommon_errors[option_id] + last_quote_errors[option_id] + rf_errors[option_id]
            #print LETTERS[option_id], total, common_good[option_id], common_errors[option_id], uncommon_errors[option_id], last_quote_errors[option_id], rf_errors[option_id]

        plan_common_good = np.zeros(shape=(14,))
        plan_common_errors = np.zeros(shape=(14,))
        plan_uncommon_errors = np.zeros(shape=(14,))
        plan_rf_errors = np.zeros(shape=(14,))
        plan_last_quote_errors = np.zeros(shape=(14,))

        for i in range(0, len(rf_answers)):

            nb_errors = 0
            for j in range(0, 7):
                if j != self.TESTING_OPTION and last_quote_answers[i, j] != real_answers[i, j]:
                    nb_errors += 1

            if nb_errors > 0:
                continue

            #length = len(self.data_train_truncated[str(int(ids[i]))])
            length = 0

            rf_answer = ''.join(str(int(x)) for x in rf_answers[i])
            last_quote_answer = ''.join(str(int(x)) for x in last_quote_answers[i])
            real_answer = ''.join(str(int(x)) for x in real_answers[i])

            if rf_answer == last_quote_answer and rf_answer == real_answer:
                plan_common_good[length] += 1

            if rf_answer == last_quote_answer and last_quote_answer != real_answer:
                plan_common_errors[length] += 1

            if rf_answer != last_quote_answer and rf_answer != real_answer and last_quote_answer != real_answer:
                plan_uncommon_errors[length] += 1

            if rf_answer != last_quote_answer and rf_answer == real_answer:
                plan_last_quote_errors[length] += 1

            if rf_answer != last_quote_answer and last_quote_answer == real_answer:
                plan_rf_errors[length] += 1

        for i in range(0, 1):
            total = plan_common_good[i] + plan_common_errors[i] + plan_uncommon_errors[i] + plan_last_quote_errors[i] + plan_rf_errors[i]
            self.logger.info('TOTAL {} {} {} {} {} {} {}'.format(i, total, plan_common_good[i], plan_common_errors[i], plan_uncommon_errors[i], plan_last_quote_errors[i], plan_rf_errors[i]))
            print 'TOTAL {} {} {} {} {} {} {}'.format(i, total, plan_common_good[i], plan_common_errors[i], plan_uncommon_errors[i], plan_last_quote_errors[i], plan_rf_errors[i])

        return plan_last_quote_errors[0] - plan_rf_errors[0]

    # Splits the data in parameter in to cv_set and train_set - the id of the subset to be used for CV is passed by the caller method
    def simple_split_cv_data(self, ids, features, last_quote_answers, real_answers, validation_subset_id, nb_subsets):

        subset_length = len(ids) / nb_subsets

        cv_subset_start = subset_length * validation_subset_id
        cv_subset_end = subset_length * (validation_subset_id + 1)

        to_train_ids = np.concatenate((ids[0:cv_subset_start], ids[cv_subset_end:]))
        to_train_features = np.concatenate((features[0:cv_subset_start], features[cv_subset_end:]))
        to_train_last_quote_answers = np.concatenate((last_quote_answers[0:cv_subset_start], last_quote_answers[cv_subset_end:]))
        to_train_real_answers = np.concatenate((real_answers[0:cv_subset_start], real_answers[cv_subset_end:]))

        to_test_ids = ids[cv_subset_start:cv_subset_end]
        to_test_features = features[cv_subset_start:cv_subset_end]
        to_test_last_quote_answers = last_quote_answers[cv_subset_start:cv_subset_end]
        to_test_real_answers = real_answers[cv_subset_start:cv_subset_end]

        return to_train_ids, to_test_ids, to_train_features, to_test_features, to_train_last_quote_answers, to_test_last_quote_answers, to_train_real_answers, to_test_real_answers

    def filter_train_data(self, ids, features, last_quote_answers, real_answers):

        tmp_ids = np.copy(ids)

        cnt_interesting = 0

        for i in range(0, len(ids)):

            nb_errors = 0
            for j in range(0, 6):
                if last_quote_answers[i, j] != real_answers[i, j]:
                    nb_errors += 1

            if nb_errors > 0:
                tmp_ids[i] = -1
            else:
                cnt_interesting += 1

        print 'Will keep', cnt_interesting, 'out of', len(ids), 'customers'

        new_ids = np.zeros(shape=(cnt_interesting,))
        new_features = np.zeros(shape=(cnt_interesting, self.NB_FEATURES))

        new_last_quote_answers = np.zeros(shape=(cnt_interesting, 7))
        new_real_answers = np.zeros(shape=(cnt_interesting, 7))

        counter = 0
        for i in range(0, len(ids)):
            if tmp_ids[i] != -1:
                new_ids[counter] = ids[i]
                new_features[counter] = features[i]
                new_last_quote_answers[counter] = last_quote_answers[i]
                new_real_answers[counter] = real_answers[i]
                counter += 1

        print len(new_ids)
        return new_ids, new_features, new_last_quote_answers, new_real_answers

    # Returns ids, features for the given data_set as well as last_quote_answers, answers
    def rf_pre_process(self, data, is_test):

        self.NB_FEATURES = 45
        ids = np.zeros(shape=(len(data),))
        features = np.zeros(shape=(len(data), self.NB_FEATURES))

        last_quote_answers = np.zeros(shape=(len(data), 7))
        answers = np.zeros(shape=(len(data), 7))

        for i, key in enumerate(data):
            ids[i] = key
            if key in self.answers:
                answers[i] = self.answers[key]

            local_features = []

            for j in range(0, 7):
                local_features.append(int(data[key][-1][self.headers['A']+j]))
                last_quote_answers[i, j] = data[key][-1][self.headers['A']+j]

            local_features.append(float(data[key][-1][self.headers['cost']]))
            local_features.append(float(data[key][-1][self.headers['cost']]) / float(data[key][-2][self.headers['cost']]))
            local_features.append(float(data[key][-1][self.headers['cost']]) / float(data[key][0][self.headers['cost']]))

            # Max/Min price
            min_price, max_price = 10000, 0
            for j in range(0, len(data[key])):
                if float(data[key][j][self.headers['cost']]) < min_price:
                    min_price = float(data[key][j][self.headers['cost']])

                if float(data[key][j][self.headers['cost']]) > max_price:
                    max_price = float(data[key][j][self.headers['cost']])

            local_features.append(float(data[key][-1][self.headers['cost']]) / min_price)
            local_features.append(max_price / float(data[key][-1][self.headers['cost']]))

            ##
            local_features.append(data[key][-1][self.headers['group_size']])
            local_features.append(data[key][-1][self.headers['homeowner']])
            local_features.append(data[key][-1][self.headers['risk_factor']])
            local_features.append(data[key][-1][self.headers['age_oldest']])
            local_features.append(data[key][-1][self.headers['duration_previous']])
            #
            local_features.append(data[key][-1][self.headers['day']])
            local_features.append(data[key][-1][self.headers['car_age']])
            local_features.append(len(data[key]))
            local_features.append(data[key][-1][self.headers['age_youngest']])
            local_features.append(data[key][-1][self.headers['married_couple']])
            local_features.append(data[key][-1][self.headers['C_previous']])
            local_features.append(data[key][-1][self.headers['location']])
            #
            local_features.append(CAR_VALUES[data[key][-1][self.headers['car_value']]])
            local_features.append(STATES[data[key][-1][self.headers['state']]])

            # Has changed some fixed statistics
            has_changed_fixed = False
            for k in range(1, len(data[key])):
                for e, header_key in enumerate(self.headers):
                    #if header_key not in LETTERS:
                    if header_key not in LETTERS and header_key != 'cost' and header_key != 'shopping_pt' and header_key != 'time' and header_key != 'day':
                        if data[key][k][self.headers[header_key]] != data[key][0][self.headers[header_key]]:
                            has_changed_fixed = True
                            break

            # How many changes between first and second quote
            change_count = 0
            for j in range(0, 7):
                if data[key][0][self.headers['A']+j] != data[key][1][self.headers['A']+j]:
                    change_count += 1

            # How many changes between first and last quote
            total_change_count = 0
            for j in range(0, 7):
                if data[key][-1][self.headers['A']+j] != data[key][0][self.headers['A']+j]:
                    total_change_count += 1

            # Nb Bigger than initial
            bigger_change_count = 0
            for j in range(0, 7):
                if data[key][-1][self.headers['A']+j] > data[key][0][self.headers['A']+j]:
                    bigger_change_count += 1

            # Nb Smaller than initial
            smaller_change_count = 0
            for j in range(0, 7):
                if data[key][-1][self.headers['A']+j] < data[key][0][self.headers['A']+j]:
                    smaller_change_count += 1

            local_features.append(smaller_change_count)
            local_features.append(bigger_change_count)

            local_features.append(has_changed_fixed)
            local_features.append(change_count)
            local_features.append(total_change_count)

            for j, val in enumerate(local_features):
                if local_features[j] == 'NA':
                    local_features[j] = 0

            # Different locations
            location_count = 1
            location_average_A_value = 0
            location_average_B_value = 0
            location_average_C_value = 0
            location_average_D_value = 0
            location_average_E_value = 0
            location_average_F_value = 0
            location_average_G_value = 0
            location_average_C_previous_value = 1
            location_G_count_value = np.zeros(shape=(5,))
            location_average_price_value = 500

            # This will not be entirely
            if data[key][0][self.headers['location']] in self.location_customers_count:
                location_count = self.location_customers_count[data[key][0][self.headers['location']]]
                if location_count > 5:

                    c_previous = 0
                    if data[key][-1][self.headers['C_previous']] != 'NA':
                        c_previous - float(data[key][-1][self.headers['C_previous']])

                    if is_test:
                        location_average_A_value = self.location_average_A[data[key][0][self.headers['location']]] / float(location_count)
                        location_average_B_value = self.location_average_B[data[key][0][self.headers['location']]] / float(location_count)
                        location_average_C_value = self.location_average_C[data[key][0][self.headers['location']]] / float(location_count)
                        location_average_D_value = self.location_average_D[data[key][0][self.headers['location']]] / float(location_count)
                        location_average_E_value = self.location_average_E[data[key][0][self.headers['location']]] / float(location_count)
                        location_average_F_value = self.location_average_F[data[key][0][self.headers['location']]] / float(location_count)
                        location_average_G_value = self.location_average_G[data[key][0][self.headers['location']]] / float(location_count)

                        location_average_C_previous_value = self.location_average_C_previous[data[key][0][self.headers['location']]] / float(location_count)
                        location_average_price_value = self.location_average_price[data[key][0][self.headers['location']]] / float(location_count)

                        for k in range(1, 5):
                            location_G_count_value[k] = self.location_G_count[data[key][0][self.headers['location']]][k] / float(location_count)

                    else:
                        location_average_A_value = (self.location_average_A[data[key][0][self.headers['location']]] - float(self.answers[key][0])) / (location_count - 1)
                        location_average_B_value = (self.location_average_B[data[key][0][self.headers['location']]] - float(self.answers[key][1])) / (location_count - 1)
                        location_average_C_value = (self.location_average_C[data[key][0][self.headers['location']]] - float(self.answers[key][2])) / (location_count - 1)
                        location_average_D_value = (self.location_average_D[data[key][0][self.headers['location']]] - float(self.answers[key][3])) / (location_count - 1)
                        location_average_E_value = (self.location_average_E[data[key][0][self.headers['location']]] - float(self.answers[key][4])) / (location_count - 1)
                        location_average_F_value = (self.location_average_F[data[key][0][self.headers['location']]] - float(self.answers[key][5])) / (location_count - 1)
                        location_average_G_value = (self.location_average_G[data[key][0][self.headers['location']]] - float(self.answers[key][6])) / (location_count - 1)

                        location_average_C_previous_value = (self.location_average_C_previous[data[key][0][self.headers['location']]] - c_previous) / (location_count - 1)
                        location_average_price_value = (self.location_average_price[data[key][0][self.headers['location']]] - float(self.data_train[key][-1][self.headers['cost']])) / (location_count - 1)

                        self.location_G_count[data[key][0][self.headers['location']]][int(self.answers[key][6])] -= 1
                        for k in range(1, 5):
                            location_G_count_value[k] = self.location_G_count[data[key][0][self.headers['location']]][k] / (location_count - 1)
                        self.location_G_count[data[key][0][self.headers['location']]][int(self.answers[key][6])] += 1

            local_features.append(location_count)
            local_features.append(location_average_A_value)
            local_features.append(location_average_B_value)
            local_features.append(location_average_C_value)
            local_features.append(location_average_D_value)
            local_features.append(location_average_E_value)
            local_features.append(location_average_F_value)
            local_features.append(location_average_G_value)
            local_features.append(location_average_C_previous_value)
            local_features.append(location_average_price_value)

            for k in range(1, 5):
                local_features.append((location_G_count_value[k]))

            if i < 40:
                print local_features

            features[i, :] = local_features

        return ids, features, last_quote_answers, answers

    def dump_answers(self, ids, answers):

        with open('rf_answers.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Customer_ID', 'Plan'])
            for i in range(0, len(ids)):
                answer = ''.join(str(int(x)) for x in answers[i])
                csv_writer.writerow([int(ids[i]), answer])

    # Train and predict
    def rf_single_option_run(self, features, real_answers, to_predict_features):
        clf = RandomForestClassifier(criterion=self.CRITERION, max_features=self.MAX_FEATURES, min_samples_leaf=self.MIN_SAMPLES_LEAF, n_estimators=self.NB_TREES, n_jobs=self.NB_JOBS)
        clf = clf.fit(features, real_answers)
        predicted_answers = clf.predict(to_predict_features)
        predicted_proba = clf.predict_proba(to_predict_features)
        return predicted_answers, predicted_proba

    # Full implementation of a simple Random Forests, run on separate options (A, B, C, D, E, F, G)
    def model_rf_separate_options(self):

        ids, features, last_quote_answers, real_answers = self.rf_pre_process(self.data_train_truncated, False)

        self.MAX_FEATURES = self.NB_FEATURES - 10
        #self.MAX_FEATURES = self.NB_FEATURES
        if self.ANALYSIS:
        # Analysing and tuning

            for i_run in range(0, self.NB_RUNS):
                total_improvement = 0  # This work only if predicting on one attribute only !

                self.MAX_FEATURES = self.NB_FEATURES - 10

                self.logger.info('Running with NB_TREES {} CRITERION {} MAX_FEATURES {} MAX_DEPTH {} MIN_SAMPLES {}'.format(self.NB_TREES, self.CRITERION, self.MAX_FEATURES, self.MAX_DEPTH, self.MIN_SAMPLES_LEAF))
                print 'Running with NB_TREES {} CRITERION {} MAX_FEATURES {} MAX_DEPTH {} MIN_SAMPLES {}'.format(self.NB_TREES, self.CRITERION, self.MAX_FEATURES, self.MAX_DEPTH, self.MIN_SAMPLES_LEAF)

                for i in range(0, self.NB_CV_SUBSETS):
                    to_train_ids, to_test_ids, to_train_features, to_test_features, to_train_last_quote_answers, to_test_last_quote_answers, to_train_real_answers, to_test_real_answers = self.simple_split_cv_data(ids, features, last_quote_answers, real_answers, i, self.NB_CV_SUBSETS)

                    self.prepare_location_stats(to_train_ids)

                    #to_train_ids, to_train_features, to_train_last_quote_answers, to_train_real_answers = self.filter_train_data(to_train_ids, to_train_features, to_train_last_quote_answers, to_train_real_answers)
                    rf_answers = np.copy(to_test_last_quote_answers)
                    # Train separately for each option
                    #for i in [0, 1, 2, 3, 4, 5, 6]:
                    for j in [self.TESTING_OPTION, ]:
                    #for j in self.TESTING_OPTIONS:
                        #print 'Predicting option', LETTERS[i]
                        #rf_answers = np.copy(to_test_last_quote_answers)

                        predicted_answers, predicted_proba = self.rf_single_option_run(to_train_features, to_train_real_answers[:, j], to_test_features)

                        #Replace only when almost sure
                        for i_id in range(0, len(to_test_ids)):
                            if max(predicted_proba[i_id]) > 0.0:
                                rf_answers[i_id, j] = predicted_answers[i_id]

                        # Check the prevision quality, and compare to the simple last known quote
                        total_improvement += self.evaluate_single_option_split(to_test_ids, rf_answers, to_test_last_quote_answers, to_test_real_answers)

                self.logger.info('IMPROVEMENT {}. {}% PARAMS NB_TREES {} CRITERION {} MAX_FEATURES {} MAX_DEPTH {} MIN_SAMPLES {}'.format(total_improvement, 100 * total_improvement / float(len(ids)), self.NB_TREES, self.CRITERION, self.MAX_FEATURES, self.MAX_DEPTH, self.MIN_SAMPLES_LEAF))
                print 'IMPROVEMENT {}. {}% PARAMS NB_TREES {} CRITERION {} MAX_FEATURES {} MAX_DEPTH {} MIN_SAMPLES {}'.format(total_improvement, 100 * total_improvement / float(len(ids)), self.NB_TREES, self.CRITERION, self.MAX_FEATURES, self.MAX_DEPTH, self.MIN_SAMPLES_LEAF)
        else:
        # Final prediction
            self.logger.info('Running FINAL with NB_TREES {} CRITERION {} MAX_FEATURES {} MAX_DEPTH {} MIN_SAMPLES {}'.format(self.NB_TREES, self.CRITERION, self.MAX_FEATURES, self.MAX_DEPTH, self.MIN_SAMPLES_LEAF))
            print 'Running FINAL with NB_TREES {} CRITERION {} MAX_FEATURES {} MAX_DEPTH {} MIN_SAMPLES {}'.format(self.NB_TREES, self.CRITERION, self.MAX_FEATURES, self.MAX_DEPTH, self.MIN_SAMPLES_LEAF)

            to_test_ids, to_test_features, to_test_last_quote_answers, to_test_real_answers = self.rf_pre_process(self.data_test, True)
            rf_answers = np.copy(to_test_last_quote_answers)
            rf_answers[:, self.TESTING_OPTION], probas = self.rf_single_option_run(features, real_answers[:, self.TESTING_OPTION], to_test_features)

            self.dump_answers(to_test_ids, rf_answers)

            return to_test_ids, rf_answers

    # -------------------------------------------------------- Main --------------------------------------------------------

    def setup_logger(self, log_filename):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        logger.handlers = []

        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger

    def run_analysis(self):
        if self.logger is None:
            self.logger = self.setup_logger('Analysis.log')
        start_time = datetime.datetime.now()
        return self.model_rf_separate_options()
        self.logger.info('All done in {}'.format(datetime.datetime.now() - start_time))
