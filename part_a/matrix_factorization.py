
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt

from starter_code.utils import *


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    rank-k approximation

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u[n] += lr * (c - u[n].T @ z[q]) * z[q]
    z[q] += lr * (c - u[n].T @ z[q]) * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 50, 100, 1000]
    for k in ks:
        re_matrix = svd_reconstruct(train_matrix, k)
        print("validation acc when k = ", k, ": ", sparse_matrix_evaluate(val_data, re_matrix))

    re_matrix = svd_reconstruct(train_matrix, 9)
    print("\nValidation acc when k = 9:", sparse_matrix_evaluate(val_data, re_matrix),
          "\n\nTest acc when k = 9:", sparse_matrix_evaluate(test_data, re_matrix), '\n')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # k = [20, 25, 30, 35, 40, 45, 48, 49, 50, 51, 52, 55, 60, 100]

    re_matrix = als(train_data, val_data, 52, 0.01, 1000000)
    # val_acc = sparse_matrix_evaluate(val_data, re_matrix)
    # print("Validation acc when k = ", 52, ":", val_acc)

    # iter_lst = np.arange(0, 1000000, 1000)
    # plt.plot(iter_lst, train_loss_lst, label="training")
    # plt.plot(iter_lst, val_loss_lst, label="validation")
    # plt.xlabel("Iterations")
    # plt.ylabel("squared-error losses")
    # plt.legend()
    # plt.show()




    total_prediction = 0
    total_accurate = 0



    val_acc = sparse_matrix_evaluate(val_data, re_matrix)
    test_acc = sparse_matrix_evaluate(test_data, re_matrix)
    print("Validation acc when k = 52: ", val_acc)
    print("\nTest acc when k = 52: ", test_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
