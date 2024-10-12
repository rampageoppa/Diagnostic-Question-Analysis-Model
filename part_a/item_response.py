import numpy as np
import matplotlib.pyplot as plt
from starter_code.utils import *


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    log_lklihood = 0.

    for i in range(len(data["user_id"])):
        theta_t = theta[data["user_id"][i]]
        beta_t = beta[data["question_id"][i]]
        c_t = data["is_correct"][i]

        log_lklihood += c_t * (theta_t - beta_t) - np.log(1. + np.exp(theta_t - beta_t))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
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
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # data的一行对应三行attr

    de_theta = np.zeros(shape=theta.shape)
    de_beta = np.zeros(shape=beta.shape)

    for i in range(len(data["user_id"])):
        theta_t = theta[data["user_id"][i]]
        beta_t = beta[data["question_id"][i]]
        c_t = data["is_correct"][i]

        de_theta[data["user_id"][i]] += c_t - sigmoid(theta_t - beta_t)
        de_beta[data["question_id"][i]] += -c_t + sigmoid(theta_t - beta_t)

    # to maximize
    theta += lr * de_theta
    beta += lr * de_beta

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []

    train_nlld = []
    val_nlld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta, beta)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        train_nlld.append(neg_lld)
        val_nlld.append(val_neg_lld)

        score = evaluate(val_data, theta, beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_nlld, val_nlld


def evaluate(data, theta, beta, ):
    """ Evaluate the model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    iteration = 50
    theta, beta, val_acc_lst, train_nlld, val_nlld = irt(train_data, val_data, 0.01, iter)
    iterations = np.arange(0, 50, 1)

    # neg-log-likelihood
    plt.plot(iterations, train_nlld, label="training")
    plt.plot(iterations, val_nlld, label="validation")
    plt.xlabel("Iterations")
    plt.ylabel("Neg-Log_Likelihood")
    plt.legend()
    plt.show()

    for i in range(len(train_nlld)):
        train_nlld[i] *= -1

    for i in range(len(val_nlld)):
        val_nlld[i] *= -1

    plt.plot(iterations, train_nlld, label="training")
    plt.plot(iterations, val_nlld, label="validation")
    plt.xlabel("Iterations")
    plt.ylabel("Log_Likelihood")
    plt.legend()
    plt.show()

    train_acc = evaluate(train_data, theta, beta)
    print("train acc for IRT model:", train_acc)
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Validation acc for IRT model:", val_acc)
    print("\nTest acc for IRT model:", test_acc)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    questions = [10, 100, 1000]
    plt.plot(np.arange(-5, 5, 0.01), sigmoid(np.arange(-5, 5, 0.01) - beta[10]), label="Question 10")
    plt.plot(np.arange(-5, 5, 0.01), sigmoid(np.arange(-5, 5, 0.01) - beta[100]), label="Question 100")
    plt.plot(np.arange(-5, 5, 0.01), sigmoid(np.arange(-5, 5, 0.01) - beta[1000]), label="Question 1000")
    for q in range(0, 100):
        plt.plot(np.arange(-5, 5, 0.01), sigmoid(np.arange(-5, 5, 0.01) - beta[q]))

    plt.xlabel("Student")
    plt.ylabel("Predicted Correct Rate")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################



if __name__ == "__main__":
    main()
