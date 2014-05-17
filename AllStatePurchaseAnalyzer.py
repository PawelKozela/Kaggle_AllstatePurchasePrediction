__author__ = 'Pawel'

import math
import csv
import numpy as np
import datetime
import logging
from sklearn.ensemble import RandomForestClassifier

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

    NB_FEATURES = 7

    NB_CV_SUBSETS = 10

    ANALYSIS = True

    # Tuning parameters
    MAX_DEPTH = 1  # NOT USED NOW - Based on min_samples_leaf
    NB_TREES = 30
    MIN_SAMPLES_LEAF = 100
    MAX_FEATURES = 20  # NOT USED NOW - All features are used
    NB_JOBS = 3

    def __init__(self, headers, data_train, data_train_truncated, data_test, answers):
        self.headers = headers
        self.data_test = data_test
        self.data_train = data_train
        self.data_train_truncated = data_train_truncated
        self.answers = answers

    def get_stat_position_bough_quote(self):

        counts = {}

        for i, key in enumerate(self.data_train):

            first_seen = 0
            last_seen = 0

            for j in range(0, len(self.data_train[key])):
                if self.data_train[key][j][self.headers['A']:self.headers['G']+1] == self.answers[key]:
                    if first_seen < 1:
                        first_seen = j+1
                    last_seen = j+1

            if (first_seen, last_seen, len(self.data_train[key])) not in counts:
                counts[(first_seen, last_seen, len(self.data_train[key]))] = 0

            counts[(first_seen, last_seen, len(self.data_train[key]))] += 1

        with open('bought_quotes_positions.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['first_seen', 'last_seen', 'length', 'count'])
            for i, key in enumerate(counts):
                    csv_writer.writerow([key[0], key[1], key[2], counts[key]])

    def get_starting_quote_with_length(self):

        counts = {}

        for i, key in enumerate(self.data_train):
            plan = "".join(tuple(self.data_train[key][0][self.headers['A']:self.headers['G']+1]))

            if plan not in counts:
                counts[plan] = {}

            if 0 not in counts[plan]:
                counts[plan][0] = 0

            if len(self.data_train[key]) not in counts[plan]:
                counts[plan][len(self.data_train[key])] = 0

            counts[plan][0] += 1

            counts[plan][len(self.data_train[key])] += 1

        with open('starting_quotes.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['plan', 'length', 'count'])
            for i, plan in enumerate(counts):
                for j, length in enumerate(counts[plan]):
                    csv_writer.writerow([plan, length, counts[plan][length]])

    # Gives the number of attribute changes between the second quote and the bought one vs the number of changes between the first and second quote
    def miss_by_changes_in_first_two_quotes(self):

        nb_miss_and_changes = np.zeros(shape=(8, 8, 15), dtype=int)

        for i, key in enumerate(self.data_train):
            real_answer = self.answers[key]
            my_answer = self.data_train[key][1][self.headers['A']:self.headers['G']+1]
            first_quote = self.data_train[key][0][self.headers['A']:self.headers['G']+1]
            second_quote = self.data_train[key][1][self.headers['A']:self.headers['G']+1]

            nb_moved = 0
            nb_changed = 0
            for j in range(0, 7):
                if first_quote[j] != second_quote[j]:
                    nb_moved += 1

                if real_answer[j] != my_answer[j]:
                    nb_changed += 1
            nb_miss_and_changes[nb_changed, nb_moved, len(self.data_train[key])] += 1

        for i in range(0, 8):
            for j in range(0, 8):
                for k in range(0, 15):
                    print i, j, k,  nb_miss_and_changes[i, j, k]

    # Gives the number of attribute changes between the second quote and the bought one vs the number of changes between the first and second quote
    def miss_by_option(self):

        nb_miss_by_option = np.zeros(shape=(8, 7), dtype=int)

        vals = np.zeros(shape=(5, 5))

        for i, key in enumerate(self.data_train_truncated):
            real_answer = self.answers[key]
            last_quote = self.data_train_truncated[key][-1][self.headers['A']:self.headers['G']+1]

            nb_changed = 0
            for j in range(0, 7):
                if real_answer[j] != last_quote[j]:
                    nb_changed += 1

            for j in range(0, 7):
                if real_answer[j] != last_quote[j]:
                    if nb_changed == 1 and j == 6:
                        vals[real_answer[j], last_quote[j]] += 1
                    nb_miss_by_option[nb_changed, j] += 1

        for i in range(1, 5):
            for j in range(1, 5):
                print i, j, vals[i, j]

        for i in range(1, 8):
            for j in range(0, 7):
                print i, LETTERS[j], nb_miss_by_option[i, j]

    # Gives the prediction accuracy of quote number x vs the total number of quotes n (ex. 40% of accuracy of quote 3 if 10 in total, 45% of quote 4 if 10 in total etc)
    def simple_stat_prediction_accuracy_by_shopping_point(self):
        good = {}
        cnt = {}
        for i, key in enumerate(self.data_train):
            purchase_len = len(self.data_train[key])
            real_answer = self.answers[key]
            for j in range(0, len(self.data_train[key])):
                my_answer = self.data_train[key][j][self.headers['A']:self.headers['G']+1]
                if (purchase_len, j) not in cnt:
                    good[(purchase_len, j)] = 0
                    cnt[(purchase_len, j)] = 0

                cnt[(purchase_len, j)] += 1
                if my_answer == real_answer:
                    good[(purchase_len, j)] += 1

        for i, key in enumerate(good):
            print key[0], key[1], cnt[key], good[key] / float(cnt[key])

    # Gives the map : bought / first / second quote option, by attribute
    def attribute_combinations(self):

        data = np.zeros(shape=(7, 6, 6, 6))
        for i, key in enumerate(self.data_train):
            real_answer = self.answers[key]
            first_quote = self.data_train[key][0][self.headers['A']:self.headers['G']+1]
            second_quote = self.data_train[key][1][self.headers['A']:self.headers['G']+1]

            for j in range(0, 7):
                data[j][int(real_answer[j])][int(first_quote[j])][int(second_quote[j])] += 1

        for i in range(0, 7):
            for j in range(0, 5):
                for k in range(0, 5):
                    for l in range(0, 5):
                        if data[i][j][k][l] > 0:
                            print ['A', 'B', 'C', 'D', 'E', 'F', 'G'][i], j, k, l, data[i][j][k][l]

    def get_next_quote(self, data):

        transitions = {}

        # To add the shopping_point_number
        for i, key in enumerate(data):

            for j in range(0, len(data[key])):
                current_plan = "".join(tuple(data[key][j][self.headers['A']:self.headers['G']+1]))
                if j < len(data[key]) - 1:
                    next_plan = "".join(tuple(data[key][j+1][self.headers['A']:self.headers['G']+1]))
                else:
                    next_plan = "".join(self.answers[key])

                if current_plan not in transitions:
                    transitions[current_plan] = {}

                if next_plan not in transitions[current_plan]:
                    transitions[current_plan][next_plan] = 0

                transitions[current_plan][next_plan] += 1

        with open('transitions.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Plan', 'Next_Plan', 'Count'])
            for i, plan in enumerate(transitions):
                for j, next_plan in enumerate(transitions[plan]):
                    csv_writer.writerow([plan, next_plan, transitions[plan][next_plan]])

        return transitions

    def get_next_quote_with_level(self, data):

        transitions = {}

        # To add the shopping_point_number
        for i, key in enumerate(data):

            for j in range(0, len(data[key])):
                current_plan = "".join(tuple(data[key][j][self.headers['A']:self.headers['G']+1]))
                if j < len(data[key]) - 1:
                    next_plan = "".join(tuple(data[key][j+1][self.headers['A']:self.headers['G']+1]))
                else:
                    next_plan = "".join(self.answers[key])

                if j not in transitions:
                    transitions[j] = {}

                if current_plan not in transitions[j]:
                    transitions[j][current_plan] = {}

                if next_plan not in transitions[j][current_plan]:
                    transitions[j][current_plan][next_plan] = 0

                transitions[j][current_plan][next_plan] += 1

        with open('transitions.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Plan', 'Next_Plan', 'Count'])
            for k, kj in enumerate(transitions):
                for i, plan in enumerate(transitions[kj]):
                    for j, next_plan in enumerate(transitions[kj][plan]):
                        csv_writer.writerow([kj, plan, next_plan, transitions[kj][plan][next_plan]])

        return transitions

    def predict(self, data, transitions):

        res = []
        res_dict = {}

        found = 0
        not_found = 0
        for i, key in enumerate(data):
            current_plan = "".join(tuple(data[key][-1][self.headers['A']:self.headers['G']+1]))

            if current_plan in transitions:
                found += 1

                # Found the one with most count
                best = 0
                best_plan = ""
                for j, next_plan in enumerate(transitions[current_plan]):
                    if transitions[current_plan][next_plan] > best:
                        best = transitions[current_plan][next_plan]
                        best_plan = next_plan

                ret = best_plan

            else:
                not_found += 1
                ret = current_plan

            res.append((key, ret))
            res_dict[key] = ret

        print len(res), found, not_found

        with open('next_quote_answers.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Customer_ID', 'plan'])
            for line_to_write in res:
                csv_writer.writerow([line_to_write[0], line_to_write[1]])

        return res_dict

    def predict_with_level(self, data, transitions):

        res = []
        res_dict = {}

        found = 0
        not_found = 0
        for i, key in enumerate(data):
            current_plan = "".join(tuple(data[key][-1][self.headers['A']:self.headers['G']+1]))

            if len(data[key])-1 in transitions and current_plan in transitions[len(data[key])-1]:
                found += 1

                # Found the one with most count
                best = 0
                best_plan = ""
                for j, next_plan in enumerate(transitions[len(data[key])-1][current_plan]):
                    if transitions[len(data[key])-1][current_plan][next_plan] > best:
                        best = transitions[len(data[key])-1][current_plan][next_plan]
                        best_plan = next_plan

                ret = best_plan

            else:
                not_found += 1
                ret = current_plan

            res.append((key, ret))
            res_dict[key] = ret

        print len(res), found, not_found

        with open('next_quote_answers.csv', 'wb') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['Customer_ID', 'plan'])
            for line_to_write in res:
                csv_writer.writerow([line_to_write[0], line_to_write[1]])

        return res_dict

    def simple_price_analysis(self, data):

        diffs = {}

        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        for k in range(0, 7):
            for i, key in enumerate(data):

                for j in range(0, len(data[key])-1):
                    #if k == 1:
                    #    print data[key][j][self.headers['A']:self.headers['G']+1]

                    current_plan = ''.join(data[key][j][self.headers['A']:self.headers['A']+k] + data[key][j][self.headers['A']+k+1:self.headers['G']+1])
                    #if k == 1:
                    #    print k, current_plan

                    next_plan = ''.join(data[key][j+1][self.headers['A']:self.headers['A']+k] + data[key][j+1][self.headers['A']+k+1:self.headers['G']+1])

                    if data[key][j][self.headers['A']+k] != data[key][j+1][self.headers['A']+k] and current_plan == next_plan:
                        price_change = int(data[key][j+1][self.headers['cost']]) - int(data[key][j][self.headers['cost']])
                        tuple_key = (letters[k], data[key][j][self.headers['A']+k], data[key][j+1][self.headers['A']+k])
                        if tuple_key not in diffs:
                            diffs[tuple_key] = []

                        diffs[tuple_key].append(price_change)

            #sns.set(style="white", palette="muted")
            #f, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=True)
            #colors = sns.color_palette("muted", 6)

        #for i, key in enumerate(diffs):
        #    print key, sum(diffs[key]) / float(len(diffs[key]))
        #    axes[int(i / 2), int(i % 2)].set_title(key)
        #    sns.distplot(diffs[key], color=colors[i], ax=axes[int(i / 2), int(i % 2)], bins=20)
        #
        #    plt.show()

            with open('simple_price_change.csv', 'wb') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Attribute', 'Current', 'Next', 'Price_change'])
                for i, key in enumerate(diffs):
                    csv_writer.writerow([key[0], key[1], key[2], sum(diffs[key]) / float(len(diffs[key]))])

    def bsf(self, data_truncated_train, data_truncated_test):
        res_dict = {}

        found = 0
        not_found = 0

        # Move to pandas

        to_analyse = []

        for i, key in enumerate(data_truncated_test):

            if i % 1000 == 0:
                print i

            if len(data_truncated_test[key]) > 2:
                res_dict[key] = ''.join(data_truncated_test[key][-1][self.headers['A']:self.headers['G']+1])
            else:
                votes = {}
                votes.clear()
                # Try to find something the same first two in the training set, and get their prediction

                for j, key2 in enumerate(data_truncated_train):
                    if data_truncated_test[key][0][self.headers['A']:self.headers['G']+1] == data_truncated_train[key2][0][self.headers['A']:self.headers['G']+1] and data_truncated_test[key][1][self.headers['A']:self.headers['G']+1] == data_truncated_train[key2][1][self.headers['A']:self.headers['G']+1]:
                        if len(data_truncated_train[key2]) > 2:
                            output = ''.join(data_truncated_train[key2][2][self.headers['A']:self.headers['G']+1])
                        else:
                            output = ''.join(self.answers[key2])

                        if output not in votes:
                            votes[output] = 0

                        votes[output] += 1

                # Didn't find shit
                if len(votes) == 0:
                    not_found += 1
                    res_dict[key] = ''.join(data_truncated_test[key][-1][self.headers['A']:self.headers['G']+1])
                else:

                    # Get the best vote
                    best_vote = ""
                    best_count = 0
                    total_count = 0

                    for j, key2 in enumerate(votes):
                        total_count += votes[key2]
                        if votes[key2] > best_count * 0.9:
                            best_vote, best_count = key2, votes[key2]

                    is_good = False
                    #if best_vote == ''.join(self.answers[key]):
                    #    is_good = True

                    is_same_as_last = False
                    if best_vote == ''.join(data_truncated_test[key][-1][self.headers['A']:self.headers['G']+1]):
                        is_same_as_last = True

                    is_last_good = False
                    #if ''.join(self.answers[key]) == ''.join(data_truncated_test[key][-1][self.headers['A']:self.headers['G']+1]):
                    #    is_last_good = True

                    to_analyse.append([best_count, total_count, math.floor(total_count / 20), float(best_count) / float(total_count), math.floor(10 * float(best_count) / float(total_count)), is_good, is_same_as_last, is_last_good])

                    if best_count > 0.5 * total_count and best_count >= 50:
                        found += 1
                        res_dict[key] = best_vote
                    else:
                        not_found += 1
                        res_dict[key] = ''.join(data_truncated_test[key][-1][self.headers['A']:self.headers['G']+1])
                    #res_dict[key] = ''.join(self.answers[key])

        with open('simple_bfs.csv', 'wb') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['Best_count', 'Total_count', 'Total_count_range', 'Pct', 'Pct_range', 'is_good', 'is_same_as_last', 'is_last_good'])
                for to_write in to_analyse:
                    csv_writer.writerow(to_write)

        print "BFS", found, not_found

        return res_dict

    def get_length_distribution(self):
        test_lengths = np.zeros((15,))
        train_lengths = np.zeros((15,))
        truncated_train_lengths = np.zeros((15,))
        for i, key in enumerate(self.data_test):
            test_lengths[len(self.data_test[key])] += 1

        for i, key in enumerate(self.data_train):
            train_lengths[len(self.data_train[key])] += 1

        for i, key in enumerate(self.data_train_truncated):
            truncated_train_lengths[len(self.data_train_truncated[key])] += 1

        for i in range(0, 15):
            print i, test_lengths[i], train_lengths[i], truncated_train_lengths[i]

    def plan_combinations(self):

        data = {}
        for i, key in enumerate(self.data_train):
            real_answer = "".join(self.answers[key])
            second_quote = "".join(tuple(self.data_train[key][-1][self.headers['A']:self.headers['G']+1]))

            if real_answer not in data:
                data[real_answer] = {}

            if second_quote not in data[real_answer]:
                data[real_answer][second_quote] = 0

            data[real_answer][second_quote] += 1

        for i, key in enumerate(data):
            for j, second_key in enumerate(data[key]):
                print key, second_key, data[key][second_key]

#  --------------------------------------------------- Random Forests ---------------------------------------------------

    # -------------------------------------------------------- Random Forests --------------------------------------------------------

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

            #length = len(self.clean_data_cv[ids[i]])
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

    # Returns ids, features for the given data_set as well as last_quote_answers, answers
    def rf_pre_process(self, data):

        self.NB_FEATURES = 46
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
                local_features.append(int(data[key][0][self.headers['A']+j]))
                local_features.append(int(data[key][-2][self.headers['A']+j]))
                local_features.append(int(data[key][-1][self.headers['A']+j]))

                last_quote_answers[i, j] = data[key][-1][self.headers['A']+j]

            local_features.append(int(data[key][-1][self.headers['cost']]))
            local_features.append(int(data[key][-1][self.headers['cost']])-int(data[key][0][self.headers['cost']]))
            #
            local_features.append(data[key][-1][self.headers['group_size']])
            local_features.append(data[key][-1][self.headers['homeowner']])
            local_features.append(data[key][-1][self.headers['risk_factor']])
            local_features.append(data[key][-1][self.headers['age_oldest']])
            local_features.append(data[key][-1][self.headers['duration_previous']])
            #
            local_features.append(data[key][-1][self.headers['car_age']])
            local_features.append(len(data[key]))
            local_features.append(data[key][-1][self.headers['age_youngest']])
            local_features.append(data[key][-1][self.headers['married_couple']])
            local_features.append(data[key][-1][self.headers['C_previous']])
            local_features.append(data[key][-1][self.headers['location']])

            local_features.append(CAR_VALUES[data[key][-1][self.headers['car_value']]])
            local_features.append(STATES[data[key][-1][self.headers['state']]])

            # Has changed some fixed statistics
            has_changed_fixed = False
            for k in range(1, len(data[key])):
                for e, header_key in enumerate(self.headers):
                    if header_key not in LETTERS:
                        if data[key][k][self.headers[header_key]] != data[key][0][self.headers[header_key]]:
                            has_changed_fixed = True
                            break

            # How many changes between 0-1
            change_count = 0
            for j in range(0, 7):
                if data[key][0][self.headers['A']+j] != data[key][1][self.headers['A']+j]:
                    change_count += 1

            total_change_count = 0
            for k in range(0, len(data[key])-1):
                for j in range(0, 7):
                    if data[key][k][self.headers['A']+j] != data[key][k+1][self.headers['A']+j]:
                        total_change_count += 1

            # G changes
            i_change_count = np.zeros(shape=(7,))
            for j in range(0, 7):
                for k in range(1, len(data[key])):
                    if data[key][k-1][self.headers['A']+j] != data[key][k][self.headers['A']+j]:
                        i_change_count[j] += 1

            for j in range(0, 7):
                local_features.append(i_change_count[j])

            local_features.append(has_changed_fixed)
            local_features.append(change_count)
            local_features.append(total_change_count)

            for j, val in enumerate(local_features):
                if local_features[j] == 'NA':
                    local_features[j] = 0

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
        clf = RandomForestClassifier(max_features=self.NB_FEATURES, min_samples_leaf=self.NB_FEATURES, n_estimators=self.NB_TREES, n_jobs=self.NB_JOBS)
        clf = clf.fit(features, real_answers)
        #export_file = tree.export_graphviz(clf.estimators_[0], out_file='tree.dot')
        predicted_answers = clf.predict(to_predict_features)
        return predicted_answers

    # Full implementation of a simple Random Forests, run on separate options (A, B, C, D, E, F, G)
    def model_rf_separate_options(self):

        timer_start_features = datetime.datetime.now()
        ids, features, last_quote_answers, real_answers = self.rf_pre_process(self.data_train_truncated)
        self.logger.info('Features done in {}'.format(datetime.datetime.now() - timer_start_features))

        if self.ANALYSIS:
        # Analysing and tuning
            total_improvement = 0 # This work only if predicting on one attribute only !
            for i in range(0, self.NB_CV_SUBSETS):
                timer_start_split = datetime.datetime.now()
                to_train_ids, to_test_ids, to_train_features, to_test_features, to_train_last_quote_answers, to_test_last_quote_answers, to_train_real_answers, to_test_real_answers = self.simple_split_cv_data(ids, features, last_quote_answers, real_answers, i, self.NB_CV_SUBSETS)
                self.logger.info('Split done in {}'.format(datetime.datetime.now() - timer_start_split))

                # Train separately for each option
                #for i in [0, 1, 2, 3, 4, 5, 6]:
                for j in [6, ]:
                    #print 'Predicting option', LETTERS[i]
                    rf_answers = np.copy(to_test_last_quote_answers)

                    timer_start_rf = datetime.datetime.now()
                    rf_answers[:, j] = self.rf_single_option_run(to_train_features, to_train_real_answers[:, j], to_test_features)
                    self.logger.info('RF done in {}'.format(datetime.datetime.now() - timer_start_rf))

                    # Check the prevision quality, and compare to the simple last known quote
                    timer_start_evaluate = datetime.datetime.now()
                    total_improvement += self.evaluate_single_option_split(to_test_ids, rf_answers, to_test_last_quote_answers, to_test_real_answers)
                    self.logger.info('Eval done in {}'.format(datetime.datetime.now() - timer_start_evaluate))

            self.logger.info('IMPROVEMENT {} out of {}. {}%'.format(total_improvement, len(ids), 100 * total_improvement / float(len(ids))))
            print 'IMPROVEMENT {} out of {}. {}%'.format(total_improvement, len(ids), 100 * total_improvement / float(len(ids)))
        else:
        # Final prediction
            to_test_ids, to_test_features, to_test_last_quote_answers, to_test_real_answers = self.rf_pre_process(self.data_test)
            rf_answers = np.copy(to_test_last_quote_answers)
            rf_answers[:, 6] = self.rf_single_option_run(features, real_answers[:, 6], to_test_features)

            self.dump_answers(to_test_ids, rf_answers)

    # -------------------------------------------------------- Main --------------------------------------------------------

    def setup_logger(self, log_filename):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        return logger

    def run_analysis(self):
        self.logger = self.setup_logger('Analysis.log')
        start_time = datetime.datetime.now()
        self.model_rf_separate_options()
        self.logger.info('All done in {}'.format(datetime.datetime.now() - start_time))
