import numpy as np
from starter_code.part_a.item_response import *


def bootstrap(data):
    user = np.asarray(data["user_id"])
    question = np.asarray(data["question_id"])
    correctness = np.asarray(data["is_correct"])

    ensemble_user = np.random.choice(user, len(user), replace=True)
    ensemble_question = np.random.choice(question,
                                         len(question), replace=True)
    ensemble_correctness = np.random.choice(correctness,
                                            len(correctness), replace=True)
    result = {
        "user_id": ensemble_user,
        "question_id": ensemble_question,
        "is_correct": ensemble_correctness
    }

    resample = np.random.choice(len(user), len(user), replace=True)
    return {
        "user_id": user[resample],
        "question_id": question[resample],
        "is_correct": correctness[resample]
    }
    # return result


def evaluate(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def bootstrap_evaluate(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        temp = []
        summation = 0
        for j in range(len(theta)):
            x = (theta[j][u] - beta[j][q]).sum()
            p = sigmoid(x)
            temp.append(p)
        for item in temp:
            summation += item
        avg_acc = summation / len(temp)
        if avg_acc >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
    total_correct = 0
    for i in range(len(pred)):
        if data["is_correct"][i] == pred[i]:
            total_correct += 1
    total = len(pred)
    return total_correct / total


if __name__ == "__main__":
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Train 3 models
    theta_1, beta_1, val_acc_lst_1, train_nlld_1, val_nlld_1 \
        = irt(bootstrap(train_data), val_data, 0.01, 100)
    theta_2, beta_2, val_acc_lst_2, train_nlld_2, val_nlld_2 \
        = irt(bootstrap(train_data), val_data, 0.01, 100)
    theta_3, beta_3, val_acc_lst_3, train_nlld_3, val_nlld_3 \
        = irt(bootstrap(train_data), val_data, 0.01, 100)

    # Simple model validation accuracy
    val_acc1 = evaluate(val_data, theta_1, beta_1)
    val_acc2 = evaluate(val_data, theta_2, beta_2)
    val_acc3 = evaluate(val_data, theta_3, beta_3)

    # Simple model test accuracy
    test_acc1 = evaluate(test_data, theta_1, beta_1)
    test_acc2 = evaluate(test_data, theta_2, beta_2)
    test_acc3 = evaluate(test_data, theta_3, beta_3)

    print("The validation accuracies of each simple model are")
    print([val_acc1, val_acc2, val_acc3])
    print("The test accuracies of each simple model are")
    print([test_acc1, test_acc2, test_acc3])

    bootstrap_theta = [theta_1, theta_2, theta_3]
    bootstrap_beta = [beta_1, beta_2, beta_3]
    final_val_acc = bootstrap_evaluate(val_data, bootstrap_theta,
                                       bootstrap_beta)
    final_test_acc = bootstrap_evaluate(test_data, bootstrap_theta,
                                        bootstrap_beta)
    print("Ensemble validation accuracy is " + str(final_val_acc))
    print("Ensemble test accuracy is " + str(final_test_acc))

