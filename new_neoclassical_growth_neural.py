import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import jsonargparse
from neoclassical_growth_benchmark import neoclassical_growth_benchmark
from typing import List, Optional


def neoclassical_growth_neural(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    k_0: float = 1.0,
    train_T: float = 30.0,
    train_points: int = 31,
    test_T: float = 40.0,
    test_points: int = 100,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    seed = 123,
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
    benchmark_grid = np.linspace(0, benchmark_T, benchmark_points)

    train = DataLoader(train_data, batch_size=len(train_data), shuffle=False)

    def derivative_back(model, t):  # backward differencing
        epsilon = 1.0e-8
        sqrt_eps = np.sqrt(epsilon)
        return (model(t)[:,:2] - model(t - sqrt_eps)[:,:2]) / sqrt_eps #takes the derivative with respect the first two elements

    # Production function
    def f(k):
        return k**a

    def f_prime(k):
        return a * (k ** (a - 1))

    def G(model, t):
        mu = model(t)[:, [0]] # mu is the 0th element
        k = model(t)[:, [1]] # k is the 1st element
        c = model(t)[:, [2]] # c is the 2nd element
        dmudt = - mu * (f_prime(k) - (delta + rho_hat))
        dkdt = f(k) - delta * k - c
        return torch.stack((dmudt, dkdt), 1).squeeze()

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
                nn.Linear(dim_hidden, 3), # mu is the 0th element, k is the 1st element, c is the 2nd element
                nn.Softplus(beta=1.0),  # To make sure capital stays positive
            )

        def forward(self, x):
            return self.q(
                x
            )  # first element is consumption, the second element is capital

    q_hat = NN()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(q_hat.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    num_epochs = 1000

    for epoch in range(num_epochs):
        for i, time in enumerate(train):
            time_zero = torch.zeros([1, 1])

            res_ode = derivative_back(q_hat, time) - G(q_hat, time)
            res_mu_dot = res_ode[:,0]
            res_k_dot = res_ode[:,1]
            res_init = q_hat(time_zero)[:, [1]] - k_0
            res_shadow = q_hat(time)[:,0]*q_hat(time)[:,2]-1.0

            loss_ode = 0.5*res_mu_dot.pow(2).mean() + 0.5*res_k_dot.pow(2).mean()
            loss_init = res_init.pow(2).mean()
            loss_shadow = res_shadow.pow(2).mean()
            loss = 0.45*loss_ode + 0.1*loss_init + 0.45*loss_shadow

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        scheduler.step()

    sol_benchmark = neoclassical_growth_benchmark(
        a, delta, rho_hat, 1.0, k_0, benchmark_grid
    )

    # Generate test_data and compare to the benchmark
    k_benchmark, c_benchmark = sol_benchmark(test_data)
    c_test = np.array(q_hat(test_data)[:, [2]].detach())
    k_test = np.array(q_hat(test_data)[:, [1]].detach())
    mu_test = np.array(q_hat(test_data)[:, [0]].detach())


    k_rel_error = np.abs(k_benchmark - k_test) / k_benchmark
    c_rel_error = np.abs(c_benchmark - c_test) / c_benchmark
    mu_rel_error = np.abs((1/c_benchmark) - mu_test) / (1/c_benchmark)

    print(
        f"E(|rel_error(k)|) = {k_rel_error.mean()}, E(|rel_error(c)|) = {c_rel_error.mean()}"
    )
    return {
        "t_train": train_data,
        "t_test": test_data,
        "k_test": k_test,
        "c_test": c_test,
        "k_benchmark": k_benchmark,
        "c_benchmark": c_benchmark,
        "k_rel_error": k_rel_error,
        "c_rel_error": c_rel_error,
        "neural_net_solution": q_hat,  # interpolator
        "benchmark_solution": sol_benchmark,  # interpolator
    }


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_neural)
