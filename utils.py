import numpy as np
import matplotlib.pyplot as plt
from svm_manual.kernel import Kernel


def choose_kernel(name):
    k = Kernel()
    if name == 'linear':
        return k.linear


def load_data(filename):
    datas = []
    labels = []
    with open(filename, 'r') as f:
        content = f.readlines()
    for line in content:
        line_list = line.strip().split(',')
        # print(line_list)
        datas.append([float(line_list[0]), float(line_list[1])])
        labels.append(float(line_list[2]))
    return np.array(datas), np.array(labels)


def plot_hyperplane(clf, x, y, h=0.02):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # SVM的分割超平面
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

