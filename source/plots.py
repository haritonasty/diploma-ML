import matplotlib.pyplot as plt
import os.path
import numpy as np

my_path = os.path.abspath(os.path.dirname(__file__))


class Numbers():
    def __init__(self):
        self.f1_score = np.array(())
        self.recall = np.array(())
        self.precision = np.array(())
        self.loss_train = np.array(())
        self.loss_test = np.array(())
        self.acc_train = np.array(())
        self.acc_test = np.array(())


def add_info(data, numbers):
    numbers.loss_train = np.append(numbers.loss_train, data[0])
    numbers.loss_test = np.append(numbers.loss_test, data[1])
    numbers.acc_train = np.append(numbers.acc_train, data[2])
    numbers.acc_test = np.append(numbers.acc_test, data[3])
    numbers.f1_score = np.append(numbers.f1_score, data[4])
    numbers.recall = np.append(numbers.recall, data[5])
    numbers.precision = np.append(numbers.precision, data[6])


def save_all_plots(numbers, name_model):
    x = [5, 10, 15, 20, 25, 30, 35, 40]
    save_acc(x, numbers.acc_train, numbers.acc_test, name_model)
    save_loss(x, numbers.loss_train, numbers.loss_test, name_model)
    print(numbers.f1_score, numbers.recall, numbers.precision)


def save_all_acc(x, f1_score, recall, precision, name_model):
    plt.figure()
    plt.xlabel('№ эпохи обучения')
    plt.ylabel('accuracy')
    plt.title('Зависимость значения метрики от № эпохи')
    plt.plot(x, f1_score)
    plt.plot(x, recall)
    plt.plot(x, precision)
    plt.legend(['f1_score', 'recall', 'precision'], loc=4)
    plt.savefig(os.path.join(my_path, '../data/plots/%s.png' % name_model))


def save_all_sets(f1_score, recall, precision, name_model):
    x = [5, 10, 15, 20, 25, 30, 35, 40]
    plt.xlabel('Количество платьев каждой девушки в обуч. выборке')
    plt.ylabel('accuracy')
    plt.title('Зависимость значения метрики от количества платьев')
    plt.plot(x, f1_score)
    plt.plot(x, recall)
    plt.plot(x, precision)
    plt.legend(['f1_score', 'recall', 'precision'], loc=4)
    plt.savefig(os.path.join(my_path, '../data/plots/sev_%s.png' % name_model))

def save_loss(x, train, test, name_model):
    plt.figure()
    plt.xlabel('Количество платьев каждой девушки в обуч. выборке')
    plt.ylabel('loss')
    plt.title('Зависимость значения метрики от количества платьев')
    plt.plot(x, train)
    plt.plot(x, test)
    plt.legend(['train', 'test'], loc=4)
    plt.savefig(os.path.join(my_path, '../data/plots/loss_%s.png' % name_model))


def save_acc(x, train, test, name_model):
    plt.figure()
    plt.xlabel('Количество платьев каждой девушки в обуч. выборке')
    plt.ylabel('accuracy')
    plt.title('Зависимость значения метрики от количества платьев')
    plt.plot(x, train)
    plt.plot(x, test)
    plt.legend(['train', 'test'], loc=4)
    plt.savefig(os.path.join(my_path, '../data/plots/acc_%s.png' % name_model))
