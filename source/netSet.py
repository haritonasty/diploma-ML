import numpy as np
import pandas as pd
import os.path

my_path = os.path.abspath(os.path.dirname(__file__))


def main():
    pass
    # net_set = generate_net_set(15)
    # save_net_set(net_set)
    # load_net_set()


class NetSet:
    def __init__(self, g_train, g_test, d_train, d_test, y_train, y_test):
        self.girls_ids_train = g_train
        self.girls_ids_test = g_test

        self.dresses_ids_train = d_train
        self.dresses_ids_test = d_test

        self.y_true_train = y_train
        self.y_true_test = y_test


def generate_test_set():
    print('generate test net_set ...')
    marks = np.array(pd.read_csv("../data/marks/matrix_of_marks.csv"))  # shape[0] -- girls, shape[1] -- dresses

    count_of_girls = marks.shape[0]
    count_of_dresses = marks.shape[1]

    girls_ids_test = np.array(())
    dresses_ids_test = np.array(())
    y_true_test = np.array(())

    for i in range(count_of_girls):
        cur_dresses = np.random.choice(np.arange(count_of_dresses), 60, replace=False)
        dresses_ids_test = np.append(dresses_ids_test, cur_dresses)
        y_true_test = np.append(y_true_test, np.take(np.array(marks[i]), cur_dresses))
        tmp = np.zeros(60)
        tmp += i + 1
        girls_ids_test = np.append(girls_ids_test, tmp).astype(int)

    pd.DataFrame({
        'girls ids': girls_ids_test,
        'dresses ids': dresses_ids_test.astype(int),
        'y true': y_true_test.astype(int)
    }).to_csv("../data/marks/different_trains/test_permanent.csv", index=False)


def generate_train_set(count_dresses_from_girl=15):
    print('generate train net_set ...')
    marks = np.array(pd.read_csv("../data/marks/matrix_of_marks.csv"))  # shape[0] -- girls, shape[1] -- dresses

    count_of_girls = marks.shape[0]
    count_of_dresses = marks.shape[1]

    girls_ids_train = np.array(())
    dresses_ids_train = np.array(())
    y_true_train = np.array(())
    mat = np.empty(shape=[count_of_girls, count_of_dresses-60])

    dresses_ids_test = np.array(
        pd.read_csv(os.path.join(my_path, "../data/marks/different_trains/test_permanent.csv"))["dresses ids"])

    print(dresses_ids_test[:60])
    print(marks[0])
    for i in range(count_of_girls):
        mat[i] = np.zeros(count_of_dresses-60)

    for i in range(count_of_girls):
        matrix_for_train = np.arange(count_of_dresses)
        mat[i] = np.delete(matrix_for_train, dresses_ids_test[(60*i):(60*(i+1))])


    for i in range(count_of_girls):
        cur_dresses = np.random.choice(mat[i], count_dresses_from_girl, replace=False).astype(int)
        dresses_ids_train = np.append(dresses_ids_train, cur_dresses)
        y_true_train = np.append(y_true_train, np.take(marks[i], cur_dresses))
        tmp = np.zeros(count_dresses_from_girl)
        tmp += i + 1
        girls_ids_train = np.append(girls_ids_train, tmp)


    pd.DataFrame({
        'girls ids': girls_ids_train.astype(int),
        'dresses ids': dresses_ids_train.astype(int),
        'y true': y_true_train.astype(int)
    }).to_csv("../data/marks/different_trains/train_" + str(count_dresses_from_girl) + ".csv", index=False)

def load_net_set(c):
    dresses_ids_train = np.array(
        pd.read_csv(os.path.join(my_path, "../data/marks/different_trains/train_" + str(c) + ".csv"))["dresses ids"])
    dresses_ids_test = np.array(
        pd.read_csv(os.path.join(my_path, "../data/marks/different_trains/test_permanent.csv"))["dresses ids"])

    girls_ids_train = np.array(
        pd.read_csv(os.path.join(my_path, "../data/marks/different_trains/train_" + str(c) + ".csv"))["girls ids"])
    girls_ids_test = np.array(
        pd.read_csv(os.path.join(my_path, "../data/marks/different_trains/test_permanent.csv"))["girls ids"])

    y_true_train = np.array(
        pd.read_csv(os.path.join(my_path, "../data/marks/different_trains/train_" + str(c) + ".csv"))["y true"])
    y_true_test = np.array(
        pd.read_csv(os.path.join(my_path, "../data/marks/different_trains/test_permanent.csv"))["y true"])

    return {
        'dresses_ids_train': dresses_ids_train,
        'dresses_ids_test': dresses_ids_test,
        'girls_ids_train': girls_ids_train,
        'girls_ids_test': girls_ids_test,
        'y_true_train': y_true_train,
        'y_true_test': y_true_test
    }


if __name__ == "__main__":
    main()
