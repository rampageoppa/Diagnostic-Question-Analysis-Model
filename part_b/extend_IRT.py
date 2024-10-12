# from starter_code.utils import *

import csv
import os
import numpy as np
import ast
import matplotlib.pyplot as plt

# path should be "/data/question_meta.csv"
# totally 388 subjects(0~377, 0 is 'math')
from starter_code.utils import load_train_csv, load_train_sparse, load_public_test_csv, \
    load_valid_csv, load_private_test_csv, save_private_test_csv


def load_question_meta(path):
    """
    load the question_meta to find numbers of different subjects of questions

    :return: the dict of different subjects
    """
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    data = {
        "question_id": [],
        "subject_id": []
    }
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))  # in order
                data["subject_id"].append(
                    ast.literal_eval(row[1]))  # made it list of list
            except ValueError:
                pass
            except IndexError:
                pass

    return data


# David
def find_largest_4(question_mata):
    """
    Find the largest 4 subject which has the most number of questions
    :param question_mata: loaded question meatdata
    :return: a list, contains the subject_id of the 4 subjects
    """
    # a frequency dict to record subject frequency
    frequency = {i: 0 for i in range(388)}  # totally 387 subjects(0 is 'math')

    for i in range(len(question_mata["question_id"])):
        subject_lst = question_mata["subject_id"][i]
        for sub in subject_lst:
            if sub != 0:  # do not append 0
                frequency[sub] += 1

    fre_sorted = sorted(frequency.items(), key=lambda item: item[1],
                        reverse=True)
    largest_4_lst = [fre_sorted[0][0], fre_sorted[1][0], fre_sorted[2][0],
                     fre_sorted[3][0]]
    return largest_4_lst


# David
def avg_correct(sub_data):
    """
    calculates the avg correctness of a particular subject
    :param sub_data: subsets of data which all belong to the same subject
    :return: float
    """
    if len(sub_data["user_id"]) == 0:
        return 0
    return sum(sub_data["is_correct"]) / len(sub_data["is_correct"])


# David
def set_K(question_meta, correct_rate):
    """
    init the k
    :param question_meta: meta
    :param correct_rate: Dict, avg correctness for each subject.
    :return: array
    """
    # length is j, which is 1774.

    # option 1: all questions' k under a subject are set to the same
    # set_K(sub_data. length)
    # k = np.array([avg_correct(sub_data) * 0.3] * length)

    sub_freq = {i: 0 for i in range(388)}
    # frequency of each subject
    for i, q in enumerate(question_meta["subject_id"]):
        for sub in q:
            if sub != 0:  # do not append 0
                sub_freq[sub] += 1

    # weight_avg is k
    weight_avg = {i: 0 for i in range(1774)}
    for i, q in enumerate(question_meta["question_id"]):
        sub_lst = question_meta["subject_id"][i]
        total = 0

        # total appearance of all subjects in a question
        for sub in sub_lst:
            total += sub_freq[sub]

        weight = 0
        for sub in sub_lst:
            weight += (sub_freq[sub] / total) * correct_rate[sub]  # rate_of_appear * avg

        weight_avg[q] = 1.0 - weight

    k = list(weight_avg.values())

    return k


# David
def find_subset(data, question_mata, subject_id):
    """
    find the subset of the data of a particular subject
    :param data:
    :param question_mata:
    :param subject_id:
    :return: subset of the data
    """
    sub_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    for i in range(len(data["user_id"])):
        q_id = data["question_id"][i]
        subject_lst = question_mata["subject_id"][q_id]

        # delete 0(math) from the lst
        if 0 in subject_lst:
            subject_lst.remove(0)

        if subject_id in subject_lst:
            sub_data["user_id"].append(data["user_id"][i])
            sub_data["question_id"].append(data["question_id"][i])
            sub_data["is_correct"].append(data["is_correct"][i])

    return sub_data


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def update_theta_beta_k(data, lr, theta, beta, K, c):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float, learning rate
    :param theta: Vector
    :param beta: Vector
    :param K: Vector
    :param c: Constant
    :return: tuple of vectors
    """
    # de_theta = np.zeros(shape=theta.shape)
    # de_beta = np.zeros(shape=beta.shape)
    # de_k = np.zeros(shape=beta.shape)

    for i in range(len(data["user_id"])):
        theta_t = theta[data["user_id"][i]]
        beta_t = beta[data["question_id"][i]]
        c_t = data["is_correct"][i]
        k_t = K[data["question_id"][i]]

        tmp = theta_t - beta_t
        tmp_e = np.exp(k_t * tmp)

        theta[data["user_id"][i]] += lr * (c_t * k_t * tmp_e / (c + tmp_e) - k_t * tmp_e / (1 + tmp_e))

        beta[data["question_id"][i]] += lr * (-c_t * k_t * tmp_e / (c + tmp_e) + k_t * tmp_e / (1 + tmp_e))

        K[data["question_id"][i]] += lr * (c_t * tmp * tmp_e / (c + tmp_e) - tmp * tmp_e / (1 + tmp_e))

    # # to maximize: +=
    # theta += lr * de_theta
    # beta += lr * de_beta
    # K += lr * de_k

    return theta, beta, K


def irt(data, val_data, lr, iterations, beta, c):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param k: Vector
    :param c: Constant
    :return: (theta, beta, K, val_acc_lst, train_acc_lst)
    """
    # Initialize theta and beta.
    theta = np.zeros(542)
    K = np.zeros(1774)

    # set K to avg correctness
    Beta = beta

    val_acc_lst = []
    train_acc_lst = []

    for i in range(iterations):
        # neg_lld = neg_log_likelihood(data, theta, beta, K, c)
        # val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        # train_nlld.append(neg_lld)
        # val_nlld.append(val_neg_lld)

        score = evaluate(val_data, theta, Beta, K, c)
        val_acc_lst.append(score)
        train_score = evaluate(data, theta, Beta, K, c)
        train_acc_lst.append(train_score)

        # print("NLLK: {} \t Score: {}".format(neg_lld, score))

        theta, beta, K = update_theta_beta_k(data, lr, theta, Beta, K, c)

    return theta, beta, K, val_acc_lst, train_acc_lst


def evaluate(data, theta, beta, K, c):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param K: Vector
    :param c: Constant
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]

        # modify: with k and c=
        x = (K[q] * (theta[u] - beta[q])).sum()
        p_a = c + (1 - c) * sigmoid(x)

        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(
        data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    private_test = load_private_test_csv("./data")



    question_meta = load_question_meta("./data/question_meta.csv")
    largest_4_lst = find_largest_4(question_meta)

    # val_loss_lst = []
    # use whole data set to train
    # for u in range(0, 150, 10):
    # option 2: use weight
    correct_rate = {i: 0 for i in range(388)}  # avg correct rate for every subject
    for i in range(1, 388):
        sub_data_tmp = find_subset(train_data, question_meta, i)
        avg_tmp = avg_correct(sub_data_tmp)
        correct_rate[i] = avg_tmp
    beta = set_K(question_meta, correct_rate)

    theta, beta, K, val_acc_lst, train_acc_lst = irt(train_data, val_data, 0.01, 100, beta, c=0.36111)  # 0.361111
    test_acc = evaluate(test_data, theta, beta, K, c=0.36111)
    val_acc = evaluate(val_data, theta, beta, K, c=0.36111)
    train_acc = evaluate(train_data, theta, beta, K, c=0.36111)
    print("\nwhole train acc when k = ", 100, ": ", train_acc)
    print("whole val acc when k = ", 100, ": ", val_acc)
    print("whole test acc when k = ", 100, ": ", test_acc)

    predictions = []

    for i, q in enumerate(private_test["question_id"]):
        u = private_test["user_id"][i]

        # modify: with k and c=
        x = (K[q] * (theta[u] - beta[q])).sum()
        p_a = 0.36111 + (1 - 0.36111) * sigmoid(x)

        if p_a >= 0.5:
            predictions.append(1.)
        else:
            predictions.append(0.)
    private_test["is_correct"] = predictions
    save_private_test_csv(private_test)



    # val_loss_lst.append(val_acc)


    # iter_lst = np.arange(0, 150, 10)
    # plt.plot(iter_lst, (0.6007620660457239, 0.7070279424216765, 0.7064634490544736, 0.7068868190798758,
    #                     0.7061812023708721, 0.7060400790290714, 0.7066045723962744, 0.7064634490544736,
    #                     0.705193338978267, 0.7058989556872707, 0.7063223257126728, 0.7053344623200677,
    #                     0.705193338978267, 0.7056167090036692, 0.7056167090036692), label="baseline model validation acc")
    # plt.plot(iter_lst, val_loss_lst, label="final model validation acc validation")
    # plt.xlabel("Iterations")
    # plt.ylabel("Validation acc")
    # plt.legend()
    # plt.show()

        # for q in range(0, 100):
        #     plt.plot(np.arange(-5, 5, 0.01), sigmoid(np.arange(-5, 5, 0.01) - beta[q]))
        # plt.xlabel("Student")
        # plt.ylabel("Predicted Correct Rate")
        # plt.legend()
        # plt.show()



    # # train each sub set separately
    # for sub in largest_4_lst:
    #     # split the original sets
    #     sub_data = find_subset(train_data, question_meta, sub)
    #     sub_val_data = find_subset(val_data, question_meta, sub)
    #     sub_test_data = find_subset(test_data, question_meta, sub)
    #
    #     # option 1
    #     # K = set_K(sub_data, 1774)
    #
    #     theta, beta, K, val_acc_lst, train_acc_lst = \
    #         irt(sub_data, sub_val_data, 0.01, 100, K, c=0.25)
    #     test_acc = evaluate(sub_test_data, theta, beta, K, c=0.25)
    #     print("subject_id:", sub, "test acc:", test_acc)


if __name__ == "__main__":
    main()
