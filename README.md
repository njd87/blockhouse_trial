# Nick Work Trial

The following are instructions and explanations for the code provided.

## impact.py

This file provides a simulated environment which follows the Obizhaeva and Wang model. If you run by default:

```console
python impact.py
```

it will render a 3x3 grid of stock prices over 100 days. The first row should all be the same fundamental stock trend.
The middle row represents how this fundamental stock trend changes based on how our one trader interacts with the market (equal trader, frequent buyer (risk prone), frequent seller (risk averse))

You can change the hyperparameters of the code with the following:

```console
python impact.py --<parameter> <value> ...
```

For example, if you want to change the resilience and time period, you would do

```console
python impact.py --time-period 1000 --resilience 0.5
```

The following are parameters you may change and the default values for them:

- starting_price: 100
- volatility: 1
- spread: 0.5
- buy_prop: 0.3333
- sell_prop: 0.3333 (NOTE: sell_prob + buy_prob <= 1)
- time_period: 100
- lmbda: 1
- shares_per_trade: 1
- resilience: 1

The code also contains estimators for the Almgren et al. model. As of right now, this function goes unused.

## RL-PPO-NICK.ipynb

This is my fork of the RL-PPO file provided. It introduces the 5 new features in the beginning and, instead of having the trader and blotter, has just the trader that has a cost function equal to that of the Obizhaeva and Wang model. The algorithm is also reactive: the price now reacts to decisions to buy and/or sell, using the observed values of AAPL stock as the fundamental trend.