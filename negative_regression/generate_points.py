import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
https://datascience.stackexchange.com/questions/120521/regression-model-doesnt-handle-negative-values
"""

alpha = 2
beta = 3
ceta = 3


def second_degree_polynomial_func(x_pts):
    y = []
    for i in x_pts:
        y.append(alpha * i * i + beta * i + ceta)
    return y


if __name__ == "__main__":
    nb_points = 10000
    absissas = np.linspace(-100, 100, nb_points)
    y = second_degree_polynomial_func(absissas)
    x = y + np.random.normal(0, alpha + beta, nb_points)

    plt.plot(absissas, y)
    plt.scatter(absissas, x, c='r')
    plt.show()

    df = pd.DataFrame(data={'X': absissas, 'y': x})
    df.to_csv("polynomial_points.csv", index=False)