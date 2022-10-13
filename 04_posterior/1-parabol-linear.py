import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, number_of_interation=100, learning_rate=0.00000000000001) -> None:
        self.number_of_interation = number_of_interation
        self.learning_rate = learning_rate
        self.w = np.array([0., 1. , 2.]).reshape(-1, 1)

    def fit(self, file_name):
        data = np.genfromtxt(file_name, delimiter=',', skip_header=1)

        N = data.shape[0]
        print(N)
        x = data[:, 0].reshape(-1, 1)
        y = data[:, 1].reshape(-1, 1)
        plt.scatter(x, y)
        plt.xlabel('mét vuông')
        plt.ylabel('giá')
        x = np.hstack((np.ones((N, 1)), x, x**2))

        cost = np.zeros((self.number_of_interation, 1))
        for i in range(1, self.number_of_interation):
            r = np.dot(x, self.w) - y
            cost[i] = 0.5*np.sum(r*r)
            self.w[0] -= self.learning_rate*np.sum(r)
            # correct the shape dimension
            self.w[1] -= self.learning_rate * \
                np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))
            self.w[2] -= self.learning_rate * \
                np.sum(np.multiply(r, x[:, 2].reshape(-1, 1)))
            print(cost[i])
        self.show(N, x)

    def show(self, N, x):
        predict = np.dot(x, self.w)

        plt.plot(x[:,1], predict, 'r')
        plt.show()

    def predict(self, square):
        price_predict = self.w[0] + self.w[1] * square
        print(f"Giá nhà cho {square} m^2 là : ", price_predict)


l = LinearRegression()
l.fit('data_square.csv')
