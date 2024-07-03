import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

"""
Parameters
-----------------
//NOTE: these are based on the paper they are found in
  these can be changed to fit the data we have
  a temporary optimization can be found in working.py
    - alpha = 1
    - beta = 3/5
    - delta = 1/4

Variables
-----------------
- I = permament impact
- J = temporary impact
- Slippage = I + J + Noise (Gaussian)
    Our goal is to provide insight into I and J via execuation times

- T = volume duration (execution time divided by number of seconds in trading day)
- X = number of shares traded
- V = average daily volume
- sigma = daily volatility
- theta = number of outstanding shares

Outputs
-----------------
- Gamma = Coefficient on parmament impact (larger magnitude = more effect)
- Eta = Coefficient on temporary impact (larger magnitude = more effect)
"""


def calculate_impact(
    slippage: list[float],
    T: list[float],
    X: list[float],
    V: list[float],
    sigma: list[float],
    theta: list[float],
    alpha: float = 1.0,
    beta: float = 0.6,
    delta: float = 0.25,
):
    """
    Calculate the impact of a trade on a stock

    Parameters
    -----------------
    T: list[float]
        The volume duration (execution time divided by number of seconds in trading day)
    X: list[float]
        The number of shares traded
    V: list[float]
        The average daily volume
    sigma: list[float]
        The daily volatility
    theta: list[float]
        The number of outstanding shares
    alpha: float
        The coefficient on permanent impact function, g
    beta: float
        The coefficient on temporary impact function, h
    delta: float
        The coefficient of liquidity

    Returns
    -----------------
    impact: list[float]
        The impact of the trade on the stock
    """
    T, X, V, sigma, theta = (
        np.array(T),
        np.array(X),
        np.array(V),
        np.array(sigma),
        np.array(theta),
    )

    # define I and J
    I = sigma * (T * np.sign(X) * np.abs(X / (V * T)) ** alpha * (theta / V) ** (delta))
    J = sigma * (np.sign(X) * np.abs(X / (V * T)) ** beta) + I / 2

    # regress slippage on I + J
    model = LinearRegression()
    model.fit(np.array([I, J]).T, slippage)

    # returns gamma, eta
    return model.coef_


class StockEnvironment:
    """
    This class is a simulation of a stock environment using the Obizhaeva and Wang model.

    Parameters
    -----------------
    starting_price: float
        The starting price of the stock
    volatility: float
        The volatility of the stock
    spread: float
        The spread of the stock
    buy_prop: float
        The proportion of time to buy
    sell_prop: float
        The proportion of time to sell
    time_period: int
        The number of time periods to simulate
    lmbda: float
        The rate at which the stock is traded relative to the density of the market
    shares_per_trade: int
        The number of shares traded per trade
    resilience: float
        The rate at which the stock returns to its fundamental price
    """

    def __init__(
        self,
        starting_price: float,
        volatility: float,
        spread: float,
        buy_prop: float,
        sell_prop: float,
        time_period: int = 23000,
        lmbda: float = 1,
        shares_per_trade: int = 1,
        resilience: float = 5,
    ):
        self.starting_price = starting_price
        self.volatility = volatility
        self.spread = spread
        self.buy_prop = buy_prop
        self.sell_prop = sell_prop
        self.time_period = time_period
        self.lmbda = lmbda
        self.shares_per_trade = shares_per_trade
        self.resilience = resilience

    def setup(self):
        """
        Set up the environment for simulation
        """
        self.steps = 0
        # set up the fundamental trend
        self.fundamental_trend = [self.starting_price]
        for _ in range(self.time_period - 1):
            self.fundamental_trend.append(
                self.fundamental_trend[-1] + np.random.normal(0, self.volatility)
            )
        self.fundamental_trend = pd.Series(self.fundamental_trend)

        # find out when buying and selling
        # this is a multinomial of 3 categories
        # 0 = hold, 1 = buy, -1 = sell
        self.actions = np.random.multinomial(
            1,
            [self.sell_prop, 1 - self.buy_prop - self.sell_prop, self.buy_prop],
            size=self.time_period,
        )
        self.actions = np.argmax(self.actions, axis=1)
        self.actions = self.actions - 1

        # the issue now is that the cumsum of self.actions up to some index i can never be negative
        # this is because we can't sell more shares than we have
        # so, fix the array such that if we try to sell more shares than we have, we just hold
        for i in range(0, len(self.actions)):
            cum_sum = np.sum(self.actions[: (i + 1)])
            if cum_sum < 0:
                self.actions[i] = 0

        self.actions = self.actions * self.shares_per_trade

        # setup place to store midpoint and ask processes
        self.midpoint = [self.starting_price]
        self.ask = [self.starting_price + self.spread / 2]

    def step(self):
        """
        Take a step in the simulation
        """
        # we now update the midpoint and ask prices
        # for this simulation, we do not consider current holdings, but rather that we are an actor that buys and sells stock
        # we assume that the upperbound of volume held is at most 1/lmbda
        self.midpoint.append(
            self.fundamental_trend[self.steps]
            + self.lmbda * np.sum(self.actions[: self.steps + 1])
        )

        # ask then is the midpoint plus half the spread plus the following sum:
        # for each action taken, we add the action * np.exp(-resilience * (steps - (time of action)))
        # this is because we assume that the stock will return to its fundamental price over time

        self.ask.append(
            self.midpoint[-1]
            + self.spread / 2
            + np.sum(
                [
                    self.actions[i] * np.exp(-self.resilience * (self.steps - i))
                    for i in range(self.steps + 1)
                ]
            )
        )

    def simulate(self):
        """
        Simulate the environment
        """
        while self.steps < self.time_period:
            self.step()
            self.steps += 1


def compare_impact():
    """
    Compare the impact of trading on a stock with different buy and sell proportions
    """
    sns.set()
    _fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    # set up the environment
    env = StockEnvironment(100, 1, 0.1, 0.3333, 0.3333, time_period=200)
    env.setup()

    # save the fundamental trend
    fundamental_trend = env.fundamental_trend

    # simulate the environment
    env.simulate()

    ax[0, 0].plot(env.fundamental_trend)
    ax[0, 0].set_title("Fundamental Trend")
    ax[1, 0].plot(env.midpoint)
    ax[1, 0].set_title("Midpoint Price, equal trade")
    ax[2, 0].plot(env.ask)
    ax[2, 0].set_title("Ask Price, equal trade")

    # set up the buy environment
    env = StockEnvironment(100, 1, 0.1, 0.6, 0.2, time_period=200)
    env.setup()

    # change the fundamental trend
    env.fundamental_trend = fundamental_trend

    # simulate the environment
    env.simulate()

    ax[0, 1].plot(fundamental_trend)
    ax[0, 1].set_title("Fundamental Trend")
    ax[1, 1].plot(env.midpoint)
    ax[1, 1].set_title("Midpoint Price, buy heavy")
    ax[2, 1].plot(env.ask)
    ax[2, 1].set_title("Ask Price, buy heavy")

    # set up the sell environment
    env = StockEnvironment(100, 1, 0.1, 0.2, 0.6, time_period=200)
    env.setup()

    env.fundamental_trend = fundamental_trend

    # simulate the environment
    env.simulate()

    ax[0, 2].plot(env.fundamental_trend)
    ax[0, 2].set_title("Fundamental Trend")
    ax[1, 2].plot(env.midpoint)
    ax[1, 2].set_title("Midpoint Price, sell heavy")
    ax[2, 2].plot(env.ask)
    ax[2, 2].set_title("Ask Price, sell heavy")

    plt.tight_layout()
    plt.show()


def plot_impact(
    starting_price: float,
    volatility: float,
    spread: float,
    buy_prop: float,
    sell_prop: float,
    time_period: int,
    lmbda: float,
    shares_per_trade: int,
    resilience: float,
):
    """
    Plot the impact of trading on a stock given the starting parameters
    """
    sns.set()

    env = StockEnvironment(
        starting_price,
        volatility,
        spread,
        buy_prop,
        sell_prop,
        time_period,
        lmbda,
        shares_per_trade,
        resilience,
    )
    env.setup()
    env.simulate()

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    ax[0].plot(env.fundamental_trend)
    ax[0].set_title("Fundamental Trend")
    ax[1].plot(env.midpoint)
    ax[1].set_title("Midpoint Price")
    ax[2].plot(env.ask)
    ax[2].set_title("Ask Price")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--starting_price",
        type=float,
        default=100,
        help="The starting price of the stock",
    )
    parser.add_argument(
        "--volatility",
        type=float,
        default=1,
        help="The volatility of the stock",
    )

    parser.add_argument(
        "--spread",
        type=float,
        default=0.1,
        help="The spread of the stock",
    )

    parser.add_argument(
        "--buy_prop",
        type=float,
        default=0.3333,
        help="The proportion of time to buy",
    )

    parser.add_argument(
        "--sell_prop",
        type=float,
        default=0.3333,
        help="The proportion of time to sell",
    )

    parser.add_argument(
        "--time_period",
        type=int,
        default=100,
        help="The number of time periods to simulate",
    )

    parser.add_argument(
        "--lmbda",
        type=float,
        default=1,
        help="The rate at which the stock is traded relative to the density of the market",
    )

    parser.add_argument(
        "--shares_per_trade",
        type=int,
        default=1,
        help="The number of shares traded per trade",
    )

    parser.add_argument(
        "--resilience",
        type=float,
        default=5,
        help="The rate at which the stock returns to its fundamental price",
    )


    args = parser.parse_args()

    # if all the args are default, run compare impact
    if all(
        [
            args.starting_price == 100,
            args.volatility == 1,
            args.spread == 0.1,
            args.buy_prop == 0.3333,
            args.sell_prop == 0.3333,
            args.time_period == 100,
            args.lmbda == 1,
            args.shares_per_trade == 1,
            args.resilience == 5,
        ]
    ):
        compare_impact()
    else:
        plot_impact(
            args.starting_price,
            args.volatility,
            args.spread,
            args.buy_prop,
            args.sell_prop,
            args.time_period,
            args.lmbda,
            args.shares_per_trade,
            args.resilience,
        )



if __name__ == "__main__":
    main()
