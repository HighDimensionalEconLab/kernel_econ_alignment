import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import jsonargparse
from asset_pricing_benchmark import mu_f_array
from typing import List, Optional


def asset_pricing_neural(
    r: float = 0.1,
    c: float = 0.02,
    g: float = -0.2,
    x_0: float = 0.01,
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50.0,
    test_points: int = 100,
    train_points_list: Optional[List[float]] = None,
    seed=123,
):
    # if passing in `train_points` then doesn't us a grid.  Otherwise, uses linspace
    if train_points_list is None:
        train_data = torch.tensor(
            np.linspace(0, train_T, train_points), dtype=torch.float32
        )
    else:
        train_data = torch.tensor(np.array(train_points_list), dtype=torch.float32)
    train_data = train_data.unsqueeze(dim=1)
    test_data = torch.tensor(np.linspace(0, test_T, test_points), dtype=torch.float32)
    test_data = test_data.unsqueeze(dim=1)

    train = DataLoader(train_data, batch_size=len(train_data), shuffle=False)

    def derivative_back(model, t):  # backward differencing
        epsilon = 1.0e-8
        sqrt_eps = np.sqrt(epsilon)
        return (model(t) - model(t - sqrt_eps)) / sqrt_eps

    # Dividends
    def x(i):
        return (x_0 + (c / g)) * np.exp(g * i) - (c / g)

    def G(model, t):
        mu = model(t)
        dmudt = r * mu - x(t)
        return dmudt

    torch.manual_seed(seed)

    class NN(nn.Module):
        def __init__(
            self,
            dim_hidden=128,
        ):
            super().__init__()
            self.dim_hidden = dim_hidden
            self.q = nn.Sequential(
                nn.Linear(1, dim_hidden, bias=True),
                nn.Tanh(),
                nn.Linear(dim_hidden, dim_hidden, bias=True),
                nn.Tanh(),
                nn.Linear(dim_hidden, dim_hidden, bias=True),
                nn.Tanh(),
                nn.Linear(dim_hidden, dim_hidden, bias=True),
                nn.Tanh(),
                nn.Linear(dim_hidden, 1),
                nn.Softplus(beta=1.0),  # To make sure price stays positive
            )

        def forward(self, x):
            return self.q(x)

    q_hat = NN()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(q_hat.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    num_epochs = 1000

    for epoch in range(num_epochs):
        for i, time in enumerate(train):

            res_ode = derivative_back(q_hat, time) - G(q_hat, time)
            res_p_dot = res_ode[:, 0]

            loss = res_p_dot.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        scheduler.step()

    # Generate test_data and compare to the benchmark
    mu_benchmark = mu_f_array(np.array(test_data), c, g, r, x_0)
    mu_test = np.array(q_hat(test_data)[:, [0]].detach())

    mu_rel_error = np.abs(mu_benchmark - mu_test) / mu_benchmark
    print(f"E(|rel_error(p)|) = {mu_rel_error.mean()}")
    return {
        "t_train": train_data,
        "t_test": test_data,
        "p_test": mu_test,
        "p_benchmark": mu_benchmark,
        "p_rel_error": mu_rel_error,
        "neural_net_solution": q_hat,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(asset_pricing_neural)
