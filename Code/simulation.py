import ast
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import random

from broker import Broker
from stock import Stock

# will have directed graph indicating how much influence
broker_network = nx.DiGraph()

# we need to pull in data over relevant intervals - maybe
stock_df = pd.read_csv('data/ticker_info_random1000.csv')
valid_tickers = stock_df['Ticker']
stock_df.set_index('Ticker', inplace=True)

# convert to dictionary of ticker to Stock class
stocks = {}
for ticker in valid_tickers:
    # compute total equity for a given stock
    try:
        dates = ast.literal_eval(stock_df.at[ticker, 'dates'])
        prices = ast.literal_eval(stock_df.at[ticker, 'time_series'])
    except ValueError:
        # print(ticker)
        pass

    forward_eps = stock_df['forwardEps'][ticker]
    earnings_growth = stock_df['earningsGrowth'][ticker]
    dividend_ratio = stock_df['dividendRate'][ticker]

    if isinstance(forward_eps, pd.Series):
        forward_eps = forward_eps[0]
        earnings_growth = earnings_growth[0]
        dividend_ratio = dividend_ratio[0]

    if math.isnan(dividend_ratio):
        dividend_ratio = 0
        earnings_growth = .05
        forward_eps = 2

    stocks[ticker] = Stock(ticker, forward_eps, earnings_growth, dividend_ratio, dates, prices)


# create list of brokers
N = 100
standard_influence = .2
time_steps = 1000
trials = 100
start = "01/01/2003"
end = "12/31/2023"

# test over 1980 to 2000 and then 2003 to 2023

# figure out what data to collect and plot from the simulation

for trial in range(trials):
    # (self, id: int, initial_money: float, risk_minimum: float=0, risk_maximum: float=math.inf,
    #                  purchase_rate_limit: int=0, neighbor_weight: float=0, shift_influence = False)
    # brokers will need to set these values based on if we're doing a influence shift or a risk variety

    brokers = [Broker(i, 1_000_000, 50, 100, 0, 0.05, False)
               for i in range(N)]

    # Generate the number of friends and populate those friend relationships with a normal distribution
    friend_count_dist = np.floor(np.array(powerlaw.Power_Law(xmin=1, parameters=[3.0]).generate_random(N)) + 2)

    """
    # Plot the histogram of the generated samples
    plt.hist(samples, bins=50, color='g', alpha=0.5, density=True)
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title('Histogram of Samples from Power Law Distribution')
    plt.show()
    """

    # create network and populate friends
    broker_network.add_nodes_from([node.id for node in brokers])
    for i in range(N):
        for j in range(int(friend_count_dist[i])):
            neighbor = np.random.normal(N/2, .34*N)
            broker_network.add_edge(i, neighbor, weight=standard_influence/friend_count_dist[i])

    # nx.draw(broker_network, with_labels=True, font_color='white', node_shape='s')
    # plt.show()

    start_date = datetime.strptime(start, "%m/%d/%Y")
    end_date = datetime.strptime(end, "%m/%d/%Y")
    date = start_date
    delta = timedelta(days=1)
    while date < end_date:
        random.shuffle(brokers)
        # TODO: replace with data for the time step, possible turn into map of stock to price/quantity available
        available_stocks = set()

        # iterate through all stocks in dataframe and add ones that exist at this time to the avialble set
        for ticker in stocks:
            stock = stocks[ticker]
            try:
                if date.strftime("%m/%d/%Y") in stock.dates:
                    available_stocks.add(ticker)
            except TypeError:
                # print(ticker)
                pass

        if len(available_stocks):
            for broker in brokers:
                broker.update(broker_network, available_stocks, brokers, stocks, date.strftime("%m/%d/%Y"))

        date += delta
        print(date.strftime("%m/%d/%Y"))

    # examine state of brokers at end
        for broker in brokers:
            print(broker.id, broker.get_status(stocks, date))


# add plotting code from whatever we decide to plot

