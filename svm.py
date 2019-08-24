import random
import numpy as np
from svm_manual import utils


class SVM:
    def __init__(self, epoch=10000, c=1.0, epsilon=0.001, kernel='linear'):
        self.w = None
        self.b = None
        self.epoch = epoch
        self.C = c
        self.epsilon = epsilon
        self.kernel = utils.choose_kernel(kernel)

    def fit(self, x, y):
        m = x.shape[0]
        alpha = np.zeros(m)
        count = 0
        while count <= self.epoch:
            count += 1
            alpha_prev = np.copy(alpha)
            for j in range(0, m):
                i = random.randint(j, m-1)
                x_i, x_j, y_i, y_j = x[i, :], x[j, :], y[i], y[j]
                eta = self.kernel(x_i, x_i) + self.kernel(x_j, x_j) - 2*self.kernel(x_i, x_j)
                if eta == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                L, H = self.compute_L_H(alpha_prime_j, alpha_prime_i, y_j, y_i)

                self.w = self.compute_w(alpha, x, y)
                self.b = self.compute_b(x, y)

                error_i = self.compute_error(x_i, y_i)
                error_j = self.compute_error(x_j, y_j)

                alpha[j] = alpha_prime_j + y_j*(error_i-error_j)/eta
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j*(alpha_prime_j - alpha[j])
            if np.linalg.norm(alpha - alpha_prev) < self.epsilon:
                break
            if count >= self.epoch:
                print(f'Iteration number exceeded {self.epoch} epoch')
        self.w = self.compute_w(alpha, x, y)
        self.b = self.compute_b(x, y)
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = x[alpha_idx, :]
        return support_vectors, count

    def compute_w(self, alpha, x, y):
        return np.dot(x.T, np.multiply(alpha, y))

    def compute_b(self, x, y):
        return np.mean(y - np.dot(x, self.w.T))

    def compute_h(self, x):
        return np.sign(np.dot(x, self.w.T) + self.b).astype(int)

    def compute_error(self, x, y):
        return self.compute_h(x) - y

    def compute_L_H(self, a_j_old, a_i_old, y_j, y_i):
        if y_i != y_j:
            # L = max(0, a2-a1), H = min(C, C+a2-a1)
            return max(0, a_j_old-a_i_old), min(self.C, a_j_old-a_i_old+self.C)
        else:
            # L = max(0, a1+a2-C) H = min(C, a1+a2)
            return max(0, a_j_old+a_i_old-self.C), min(self.C, a_j_old+a_i_old)

    def predict(self, x):
        return self.compute_h(x)

    def accuracy(self, y, y_hat):
        idx = np.where(y_hat == 1)
        TP = np.sum(y_hat[idx] == y[idx])
        idx = np.where(y_hat == -1)
        TN = np.sum(y_hat[idx] == y[idx])
        return float(TP + TN) / len(y)
