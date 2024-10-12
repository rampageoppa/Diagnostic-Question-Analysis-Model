from sklearn.impute import KNNImputer
from starter_code.utils import *
import matplotlib.pyplot as plt
import numpy as np

from starter_code.utils import load_public_test_csv, load_train_sparse, \
    load_valid_csv, sparse_matrix_evaluate


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Acc when k = ", k,": {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy when k = ", k,": {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # print("Sparse matrix:")
    # print(sparse_matrix)
    # print("Shape of sparse matrix:")
    # print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # Compute the accuracy of each value of k and plot the relationship
    # (User based)
    print("User Based:")
    list_k = [1, 6, 11, 16, 21, 26]
    accuracy = []
    for k in list_k:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracy.append(acc)
    plt.xlabel("values of k")
    plt.ylabel("accuracy")
    plt.title("Relation between the value of k and accuracy (User based)")
    plt.plot(list_k, accuracy)
    # plt.savefig("User k-accuracy relation")
    plt.show()

    # Select the value of k that maximize accuracy (User based)
    acc = np.asarray(accuracy)
    max_index = np.argmax(acc)
    k_star = list_k[max_index]
    best_acc = accuracy[max_index]
    print("Test accuracy: ")
    testing_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print("Validation acc when k = 11: " + str(best_acc))
    print("\nTest acc when k = 11 " + str(testing_acc))

    # Compute the accuracy of each value of k and plot the relationship
    # (Item based)
    print("Item Based:")
    list_k = [1, 6, 11, 16, 21, 26]
    accuracy = []
    for k in list_k:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        accuracy.append(acc)
    plt.xlabel("values of k")
    plt.ylabel("accuracy")
    plt.title("Relation between the value of k and accuracy (Item based)")
    plt.plot(list_k, accuracy)
    # plt.savefig("Item k-accuracy relation")
    plt.show()

    # Select the value of k that maximize accuracy (Item based)
    acc = np.asarray(accuracy)
    max_index = np.argmax(acc)
    k_star = list_k[max_index]
    best_acc = accuracy[max_index]
    print("Test accuracy: ")
    testing_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print("Validation acc when k = 21: " + str(best_acc))
    print("\nTest acc when k = 21: " + str(testing_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
