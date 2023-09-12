import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.distributed._shard.checkpoint import load_state_dict

from negative_regression.generate_points import second_degree_polynomial_func

if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )

    df = pd.read_csv("polynomial_points.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.X.values, df.y.values, test_size=0.33, random_state=42,
                                                        shuffle=True)
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    model.load_state_dict(torch.load("model_poly.pt"))

    model.eval()
    # print(model(torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1,1)))
    with torch.no_grad():

        plt.plot(np.sort(X_test.squeeze()), second_degree_polynomial_func(np.sort(X_test.squeeze())), c='r')
        plt.scatter(X_test, model(X_test).detach().numpy())
        plt.show()

