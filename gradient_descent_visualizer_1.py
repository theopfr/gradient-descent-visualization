
from preprocessing import load_csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

plt.style.use("ggplot")


class Plot:
    def __init__(self, dataset_path: str="", amount: int=1000, epochs: int=10000, lr: float=0.00001):
        self.dataset = load_csv(dataset_path)
        self.amount = amount

        self.epochs = epochs
        self.epoch = 0
        self.lr = lr

        self.theta_0 = random.uniform(-1, 1)  
        self.theta_1 = random.uniform(-1, 1)  

        self.x = (v := np.array(self.dataset["GrLivArea"].tolist()[:self.amount])) / max(v)
        self.y = (w := np.array(self.dataset["SalePrice"].tolist()[:self.amount])) / max(w)

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax = self.fig.add_subplot(1, 2, 2, projection="3d")

    # mean squared error
    def MSE(self, y_prediction, y_true, deriv=(False, 1)):
        if deriv[0]:
            # deriv[1] is the  derivitive of the fit_function
            return 2 * np.mean(np.subtract(y_true, y_prediction) * -deriv[1])
        return np.mean(np.square(np.subtract(y_true, y_prediction)))

    # linear function
    def fit_function(self, t0, t1, x):
        return t0 + (t1 * x)

    # get loss for every theta_0/_1
    def get_loss_points(self):
        theta_range = np.arange(-1, 1, 0.1)

        loss_matrix = np.ones((len(theta_range), len(theta_range)))
        xs = np.ones((len(theta_range), len(theta_range)))
        ys = np.ones((len(theta_range), len(theta_range)))

        for i, t0 in enumerate(theta_range):
            for j, t1 in enumerate(theta_range):
                output = self.fit_function(t0, t1, self.x)
                loss = self.MSE(output, self.y)

                xs[i][j] = t0
                ys[i][j] = t1

                loss_matrix[i][j] = loss

        return xs, ys, loss_matrix

    # train model
    def train(self, i):
        self.epoch += 1

        xs, ys, zs = self.get_loss_points()

        #for epoch in range(epochs):
        predictions = self.fit_function(self.theta_0, self.theta_1, self.x)
        loss = self.MSE(predictions, self.y)
        
        # gradient descent
        delta_theta_0 = self.MSE(predictions, self.y, deriv=(True, 1))
        delta_theta_1 = self.MSE(predictions, self.y, deriv=(True, self.x))

        self.theta_0 -= self.lr * delta_theta_0
        self.theta_1 -= self.lr * delta_theta_1

        # print loss and plot live time plot
        print("\nepoch", self.epoch, ", loss:", loss)

        self.ax.clear()
        self.ax.plot_surface(xs, ys, zs, cmap="plasma", alpha=0.70)
        self.ax.scatter(self.theta_1, self.theta_0, loss, s=150, zorder=1)
        
        self.ax1.clear()
        variables, predictions = map(list, zip(*sorted(zip(self.x, predictions))))
        self.ax1.plot(variables, predictions, "b")
        self.ax1.scatter(self.x, self.y, c="r")

    # live plot
    def visualize(self):
        ani = FuncAnimation(self.fig, self.train, frames=2, interval=0.01)
        plt.tight_layout()
        plt.show()


plot = Plot(dataset_path="dataset/houston_housing/single_variable_dataset/train.csv",
            epochs=150000,
            lr=0.00075,
            amount=500)

plot.visualize()



