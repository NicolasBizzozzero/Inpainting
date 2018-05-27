
import os.path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso

from src.linear.linear_regression import LinearRegression


PATH_DIR_USPS = os.path.join(os.path.dirname(__file__), "../res/USPS")
PATH_USPS_TRAIN = os.path.join(PATH_DIR_USPS, "USPS_train.txt")
PATH_USPS_TEST = os.path.join(PATH_DIR_USPS, "USPS_test.txt")


def test_all_usps_1_vs_all(classifieur, **kwaargs):
    np.set_printoptions(threshold=np.nan)  # Print all array

    print("Fonction de coût :", kwaargs["loss_g"].__name__)
    results = np.zeros((10,))
    for pos in range(10):
        clsf = classifieur(**kwaargs)
        res = test_usps_1_vs_all(class_pos=pos, classifieur=clsf)
        print("pos:", pos, "score:", res)
        results[pos] = res
        print("W :", clsf.w)
        print("nb 0 :", len(clsf.w[clsf.w == 0]))
        print("mean w :", np.mean(np.abs(clsf.w)))
        print("___")
    results = np.array(results).reshape((1, 10))
    print(results)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(results, ax=ax, vmin=0.0, vmax=1.0)
    plt.yticks([], [])
    plt.savefig("1_vs_all_" + kwaargs["loss_g"].__name__ + "_" + str(kwaargs["alpha"]) + ".png")
    plt.close("all")
    # plt.show()


def test_usps_1_vs_all(class_pos, classifieur):
    datax_train, datay_train, datax_test, datay_test = load_usps_1_vs_all(
        class_pos)
    classifieur.fit(datax_train, datay_train)
    return classifieur.score(datax_test, datay_test)


def test_all_usps(classifieur, **kwaargs):
    np.set_printoptions(threshold=np.nan)  # Print all array

    results = np.zeros((10, 10))
    for neg in range(10):
        for pos in range(neg + 1, 10):
            print("neg=" + str(neg) + ", pos=" + str(pos))
            clf = classifieur(**kwaargs)
            results[neg, pos] = test_usps(class_neg=neg, class_pos=pos,
                                          classifieur=clf)
            print("Result :", results[neg, pos])
            print("W :", clf.w)
            print("nb 0 :", len(clf.w[clf.w == 0]))
            print("mean w :", np.mean(np.abs(clf.w)))
            print("___")
    sns.heatmap(results, vmin=0.0, vmax=1.0)
    # plt.show()
    plt.savefig("all_vs_all_" + kwaargs["loss_g"].__name__ + "_" + str(kwaargs["alpha"]) + ".png")
    plt.close("all")


def test_all_usps_sklearn(alpha):
    np.set_printoptions(threshold=np.nan)  # Print all array

    results = np.zeros((10, 10))
    for neg in range(10):
        for pos in range(neg + 1, 10):
            print("neg=" + str(neg) + ", pos=" + str(pos))
            clf = Lasso(alpha=alpha)
            datax_train, datay_train, datax_test, datay_test = load_usps(neg, pos)
            clf.fit(datax_train, datay_train)
            score = (np.sign(clf.predict(datax_test)) == datay_test)
            score = len(score[score == True]) / len(score)
            results[neg, pos] = score
            print("Result :", results[neg, pos])
            print("W :", clf.coef_)
            print("nb 0 :", len(clf.coef_[clf.coef_ == 0]))
            print("mean w :", np.mean(np.abs(clf.coef_)))
            print("___")
    sns.heatmap(results, vmin=0.0, vmax=1.0)
    # plt.show()
    plt.savefig("all_vs_all_" + "sklearnlasso" + ".png")
    plt.close("all")

    from sklearn.linear_model import LinearRegression
    results = np.zeros((10, 10))
    for neg in range(10):
        for pos in range(neg + 1, 10):
            print("neg=" + str(neg) + ", pos=" + str(pos))
            clf = LinearRegression()
            datax_train, datay_train, datax_test, datay_test = load_usps(neg, pos)
            clf.fit(datax_train, datay_train)
            score = (np.sign(clf.predict(datax_test)) == datay_test)
            score = len(score[score == True]) / len(score)
            results[neg, pos] = score
            print("Result :", score)
            print("W :", clf.coef_)
            print("nb 0 :", len(clf.coef_[clf.coef_ == 0]))
            print("mean w :", np.mean(np.abs(clf.coef_)))
            print("___")
    sns.heatmap(results, vmin=0.0, vmax=1.0)
    # plt.show()
    plt.savefig("all_vs_all_" + "sklearnlinearregression" + ".png")
    plt.close("all")


def test_all_usps_1_vs_all_sklearn():
    np.set_printoptions(threshold=np.nan)  # Print all array

    results = np.zeros((10,))
    for pos in range(10):
        clsf = Lasso()
        datax_train, datay_train, datax_test, datay_test = load_usps_1_vs_all(pos)
        clsf.fit(datax_train, datay_train)
        score = (np.sign(clsf.predict(datax_test)) == datay_test)
        score = len(score[score == True]) / len(score)
        print("pos:", pos, "score:", score)
        results[pos] = score
        print("W :", clsf.coef_)
        print("nb 0 :", len(clsf.coef_[clsf.coef_ == 0]))
        print("mean w :", np.mean(np.abs(clsf.coef_)))
        print("___")
    results = np.array(results).reshape((1, 10))
    print(results)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(results, ax=ax, vmin=0.0, vmax=1.0)
    plt.yticks([], [])
    plt.savefig("1_vs_all_" + "lasso" + ".png")
    plt.close("all")

    from sklearn.linear_model import LinearRegression
    results = np.zeros((10,))
    for pos in range(10):
        clsf = LinearRegression()
        datax_train, datay_train, datax_test, datay_test = load_usps_1_vs_all(pos)
        clsf.fit(datax_train, datay_train)
        score = (np.sign(clsf.predict(datax_test)) == datay_test)
        score = len(score[score == True]) / len(score)
        print("pos:", pos, "score:", score)
        results[pos] = score
        print("W :", clsf.coef_)
        print("nb 0 :", len(clsf.coef_[clsf.coef_ == 0]))
        print("mean w :", np.mean(np.abs(clsf.coef_)))
        print("___")
    results = np.array(results).reshape((1, 10))
    print(results)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.heatmap(results, ax=ax, vmin=0.0, vmax=1.0)
    plt.yticks([], [])
    plt.savefig("1_vs_all_" + "linearregression" + ".png")
    plt.close("all")


def test_usps(class_neg, class_pos, classifieur):
    datax_train, datay_train, datax_test, datay_test = load_usps(class_neg,
                                                                 class_pos)
    classifieur.fit(datax_train, datay_train)
    return classifieur.score(datax_test, datay_test)


def load_usps(class_neg, class_pos, filename_train=PATH_USPS_TRAIN,
              filename_test=PATH_USPS_TEST, label_neg=-1, label_pos=1):
    with open(filename_train, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    datax_train, datay_train = tmp[:, 1:], tmp[:, 0].astype(int)
    indexes_neg = np.where(datay_train == class_neg)[0]
    indexes_pos = np.where(datay_train == class_pos)[0]
    datax_train = np.vstack((datax_train[indexes_neg],
                             datax_train[indexes_pos]))
    datay_train = np.hstack((np.ones((len(indexes_neg),)) * label_neg,
                             np.ones((len(indexes_pos),)) * label_pos))

    with open(filename_test, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    datax_test, datay_test = tmp[:, 1:], tmp[:, 0].astype(int)
    indexes_neg = np.where(datay_test == class_neg)[0]
    indexes_pos = np.where(datay_test == class_pos)[0]
    datax_test = np.vstack((datax_test[indexes_neg],
                            datax_test[indexes_pos]))
    datay_test = np.hstack((np.ones((len(indexes_neg),)) * label_neg,
                            np.ones((len(indexes_pos),)) * label_pos))

    datax_train, datay_train = _shuffle_data(datax_train, datay_train)
    datax_test, datay_test = _shuffle_data(datax_test, datay_test)
    return datax_train, datay_train, datax_test, datay_test


def load_usps_1_vs_all(class_pos, filename_train=PATH_USPS_TRAIN,
                       filename_test=PATH_USPS_TEST, label_neg=-1,
                       label_pos=1):
    with open(filename_train, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    datax_train, datay_train = tmp[:, 1:], tmp[:, 0].astype(int)
    indexes_neg = np.where(datay_train != class_pos)[0]
    indexes_pos = np.where(datay_train == class_pos)[0]
    datax_train = np.vstack((datax_train[indexes_neg],
                             datax_train[indexes_pos]))
    datay_train = np.hstack((np.ones((len(indexes_neg),)) * label_neg,
                             np.ones((len(indexes_pos),)) * label_pos))

    with open(filename_test, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split()) > 2]
    tmp = np.array(data)
    datax_test, datay_test = tmp[:, 1:], tmp[:, 0].astype(int)
    indexes_neg = np.where(datay_test != class_pos)[0]
    indexes_pos = np.where(datay_test == class_pos)[0]
    datax_test = np.vstack((datax_test[indexes_neg],
                            datax_test[indexes_pos]))
    datay_test = np.hstack((np.ones((len(indexes_neg),)) * label_neg,
                            np.ones((len(indexes_pos),)) * label_pos))

    datax_train, datay_train = _shuffle_data(datax_train, datay_train)
    datax_test, datay_test = _shuffle_data(datax_test, datay_test)
    return datax_train, datay_train, datax_test, datay_test


def _shuffle_data(datax, datay):
    """ Bouge aléatoirement la position de toutes les données tout en
    conservant l'ordre des exemples et des labels.
    """
    permutations = np.random.permutation(len(datax))
    return datax[permutations], datay[permutations]


if __name__ == "__main__":
    pass
