
from preprocessing import load_csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("ggplot")


# mean squared error
def MSE(y_prediction, y_true, deriv=(False, 1)):
    if deriv[0]:
        # deriv[1] is the  derivitive of the fit_function
        return 2 * np.mean(np.subtract(y_true, y_prediction) * -deriv[1])
    return np.mean(np.square(np.subtract(y_true, y_prediction)))

# linear function
def fit_function(theta_0, theta_1, x):
    return theta_0 + (theta_1 * x)

# learning rate decay (not in usage)
def lr_decay(current_lr, decay_rate, epoch, periode=15):
    if epoch % periode == 0:
        return current_lr * (1 / (1 + decay_rate*epoch))
    else:
        return current_lr

# get loss for every theta_0/_1
def get_loss_points(dataset: list):
    theta_range = np.arange(-1, 1, 0.1)

    x, y = dataset[0], dataset[1]

    loss_matrix = np.ones((len(theta_range), len(theta_range)))
    xs = np.ones((len(theta_range), len(theta_range)))
    ys = np.ones((len(theta_range), len(theta_range)))

    for i, theta_0 in enumerate(theta_range):
        for j, theta_1 in enumerate(theta_range):
            output = fit_function(theta_0, theta_1, x[0])
            loss = MSE(output, y[0])

            xs[i][j] = theta_0
            ys[i][j] = theta_1

            loss_matrix[i][j] = loss

    return xs, ys, loss_matrix

# plot loss surface curve
def plot_loss_surface(x, y, z, point=(0, 0, 0)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x, y, z, cmap="plasma")
    ax.scatter(point[0], point[1], point[2], s=150, zorder=1)

    plt.show()

# train model
def train(dataset, epochs=10, lr=0.01):
    # loading and normalizing the data
    x = (v := np.array(dataset["GrLivArea"].tolist()[:500])) / max(v)
    y = (l := np.array(dataset["SalePrice"].tolist()[:500])) / max(l)

    # initializing random theta starting values
    # theta_0 = random.uniform(-1, 1)                                   # y-intercept
    # theta_1 = random.uniform(-1, 1)                                   # slope

    # or start at highest loss (uncomment below, comment above)
    theta_0 = -1                                                        # y-intercept
    theta_1 = -1                                                        # slope

    # plot loss surface and starting point (theta_0, theta_1)
    xs, ys, zs = get_loss_points((x, y))
    plot_loss_surface(xs, ys, zs, point=(theta_0, theta_1, MSE(fit_function(theta_0, theta_1, x[0]), y[0])))

    # train
    for epoch in range(epochs):
        predictions = fit_function(theta_0, theta_1, x)
        loss = MSE(predictions, y)
        
        # gradient descent
        delta_theta_0 = MSE(predictions, y, deriv=(True, 1))
        delta_theta_1 = MSE(predictions, y, deriv=(True, x))

        theta_0 -= lr * delta_theta_0
        theta_1 -= lr * delta_theta_1

        # print loss and plot live time plot
        if epoch % 750 == 0:
            print("\nepoch:", epoch, ", loss:", loss, " , lr:", lr)
            plt.clf()

            variables, predictions = map(list, zip(*sorted(zip(x, predictions))))
            plt.plot(variables, predictions, "b")
            plt.scatter(x, y, c="r")

            plt.pause(0.01)

    plt.show()

    # plot loss surface and the optimal point (theta_0, theta_1)
    plot_loss_surface(xs, ys, zs, point=(theta_1, theta_0, np.mean(predictions)))


    
train(load_csv("dataset/houston_housing/single_variable_dataset/train.csv"), epochs=150000, lr=0.00075)




