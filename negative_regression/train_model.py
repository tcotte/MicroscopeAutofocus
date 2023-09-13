import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Define the model
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import optim
from tqdm import tqdm

model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)


# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.001)


df = pd.read_csv("polynomial_points.csv")
normalized_data = preprocessing.normalize([df.X.values])

X_train, X_test, y_train, y_test = train_test_split(np.squeeze(normalized_data), df.y.values, test_size=0.33, random_state=42, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)



# Hold the best model
best_mse = np.inf  # init to infinity
best_weights = None
history = []

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

if __name__ == "__main__":

    device = "cpu"

    model.to(device)
    y_test.to(device)

    for epoch in range(n_epochs):
        model.train()
        with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start + batch_size]
                y_batch = y_train[start:start + batch_size]

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        model.to(device)
        X_test.to(device)

        y_pred = model(X_test)

        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch} -- MSE {mse} -- Best MSE {best_mse}")

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    torch.save(model.state_dict(), "model_poly_norm.pt")
    plt.plot(history)
    plt.show()