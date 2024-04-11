import ast
from datetime import datetime, timedelta
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import random
from time import time

from broker import Broker
from stock import Stock

# pull data from csv file of stock data
filename = 'data/ticker_info_400_all_catagories.csv'
print(f"Reading pandas dataframe of stocks at {filename} and converting to dictionary of Stock at {time()}")
stock_df = pd.read_csv(filename)

# permit indexing of tickers
valid_tickers = stock_df['Ticker']
stock_df.set_index('Ticker', inplace=True)

# convert to dictionary of ticker to Stock class
stocks = {}
for idx, ticker in enumerate(valid_tickers):
    try:
        dates = ast.literal_eval(stock_df.at[ticker, 'dates'])
        prices = ast.literal_eval(stock_df.at[ticker, 'time_series'])
    except ValueError:
        # print(ticker)
        continue

    forward_eps = stock_df['forwardEps'][ticker]
    earnings_growth = stock_df['earningsGrowth'][ticker]
    dividend_ratio = stock_df['dividendRate'][ticker]

    if isinstance(forward_eps, pd.Series):
        forward_eps = forward_eps.iloc[0]
        earnings_growth = earnings_growth.iloc[0]
        dividend_ratio = dividend_ratio.iloc[0]
        print("HERE")

    stocks[ticker] = Stock(ticker, forward_eps, earnings_growth, dividend_ratio, dates, prices)
    if idx % 100 == 0:
        print(f"completed ticker {ticker} at index {idx} at time {time()}...")
    if idx > 150:  # TODO: REMOVE WHEN ACTUALLY RUNNING
        break  # cut off sooner to speed up integration testing

print(f"stocks dictionary created successfully ({time()})...")


# make directed graph for influence
broker_network = nx.DiGraph()


# create list of brokers
N = 100
standard_influence = .2
time_steps = 1000
trials = 1
start = "01/01/2003"
end = "12/31/2003"

# test over 1980 to 2000 and then 2003 to 2023

# figure out what data to collect and plot from the simulation

for trial in range(trials):
    # stats per trial
    broker_statuses = defaultdict(lambda: list())
    broker_risks = defaultdict(lambda: list())
    dates = list()  # for labels

    # brokers will need to set these values based on if we're doing a influence shift or a risk variety

    brokers = [Broker(i, 1_000_000, (i+1)/N*15, (i+1)/N*25, 0, 0.05, True)
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
            neighbor = abs(math.ceil(np.random.normal(N/2, .34*N))) % 100
            if neighbor != i:
                broker_network.add_edge(i, neighbor, weight=standard_influence/friend_count_dist[i])
                brokers[neighbor].in_neighbors[i] = standard_influence / friend_count_dist[i]  # not a clean way to do this but it'll work hopefully

    # display the network created
    # nx.draw(broker_network, with_labels=True, font_color='white', node_shape='s')
    # plt.show()

    start_date = datetime.strptime(start, "%m/%d/%Y")
    end_date = datetime.strptime(end, "%m/%d/%Y")
    date = start_date
    delta = timedelta(days=1)
    curr_time = time()
    while date < end_date:
        print(date.strftime("%m/%d/%Y"), f'-------{time()-curr_time}-------------------------------------')
        curr_time = time()
        dates.append(date.strftime('%m/%d/%Y'))

        random.shuffle(brokers)
        available_stocks = set()

        # iterate through all stocks in dataframe and add ones that exist at this time to the available set
        for ticker in stocks:
            stock = stocks[ticker]
            try:
                if date.strftime("%m/%d/%Y") in stock.date_to_price and stock.date_to_price[date.strftime("%m/%d/%Y")] > 0.00001:
                    available_stocks.add(ticker)
            except TypeError:
                # print(ticker)
                pass

        if len(available_stocks):
            prev = time()
            for i, broker in enumerate(brokers):
                # broker.money += 1_000
                broker.update(broker_network, available_stocks, brokers, stocks, date.strftime("%m/%d/%Y"))
                # print(f'', time()-prev, f'time {i}')
                prev = time()

        for broker in brokers:
            broker_statuses[broker.id].append(broker.get_status(stocks, date.strftime("%m/%d/%Y")))
            broker_risks[broker.id].append(broker.current_risk)
            if broker_statuses[broker.id][-1] < 0:
                print(broker.id, broker.get_status(stocks, date.strftime("%m/%d/%Y")), broker.money, broker.current_risk, broker.portfolio)

        date += delta

    # fix all the broker network edges
    for b in brokers:
        for n in b.in_neighbors:
            broker_network.edges[n, b.id]['weight'] = b.in_neighbors[n]


    # add plotting code from whatever we decide to plot

    # draw final state of the influence network
    pos = nx.spring_layout(broker_network, seed=3)
    nx.draw_networkx(broker_network, pos)
    print(broker_network.edges(data='weight'))
    for edge in broker_network.edges(data='weight'):
        nx.draw_networkx_edges(broker_network, pos, edgelist=[edge], width=edge[2]*2)
    plt.show()

    # compute average value of brokers
    total_wealth = 0
    for broker in broker_statuses:
        total_wealth += broker_statuses[broker][-1]
    print(f'beginning wealth: 1_000_000')
    print(f'average ending wealth: {total_wealth/N}')

    def plot_time(x, y, xlabel, ylabel, title, filename):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x, y)
        # Customize the plot
        plt.title(title)
        tick_locations = [len(x)//7*i for i in range(8)]
        labels = [x[t] for t in tick_locations[:-1]]
        labels.append(x[-1])
        plt.xticks(tick_locations, labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures/{filename}')
        plt.show()

    def plot_times(x, ys, serieslabels, xlabel, ylabel, title, filename):
        plt.figure(figsize=(10, 6))
        for i in range(len(ys)):
            plt.plot(x, ys[i], label=f'{serieslabels[i]}')

        tick_locations = [len(x)//7*i for i in range(8)]
        labels = [x[t] for t in tick_locations[:-1]]
        labels.append(x[-1])
        plt.xticks(tick_locations, labels)

        # Customize the plot
        plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures/{filename}')
        plt.show()


    def plot(x, y, xlabel, ylabel, title, filename):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o')
        plt.xlabel(xlabel), plt.ylabel(ylabel), plt.title(title)
        plt.grid(True)
        plt.savefig(f'figures/{filename}')
        plt.show()


    broker_ids = list()
    num_stocks = list()
    num_unique_stocks = list()
    cash_at_end = list()
    for b in brokers:
        broker_ids.append(b.id)
        count = 0
        unique_count = 0
        for p in b.portfolio:
            count += b.portfolio[p]
            if b.portfolio[p] > 1:
                unique_count += 1
        num_stocks.append(count)
        num_unique_stocks.append(unique_count)
        cash_at_end.append(b.money)

    # plot statistics of portfolio at end
    plot(broker_ids, num_stocks, 'Broker ids as ordered by risk level', 'number of stocks owned',
         'Number of stocks owned to broker id/risk minimum', 'stocksOwnedToBrokerIds.png')
    plot(broker_ids, num_unique_stocks, 'Broker ids as ordered by risk level', 'number of unique stocks owned',
         'Number of unique stocks owned to broker id', 'uniqueStocksOwnedToBrokerIds.png')
    plot(broker_ids, cash_at_end, 'Broker ids as ordered by risk level', 'Liquid currency ($)',
         f'Liquid currency at end of simulation {end}', f'liquidCurrency.png')


    # plot some time series of brokers net worth - might overlay these on the same plots actually
    for b in random.sample(brokers, 1):
        plot_time(dates, broker_statuses[b.id], 'dates', f'portfolio and currency value of broker {b.id}', f'broker wealth time series {b.id}', f'timeseries{b.id}.png')
        plot_time(dates, broker_risks[b.id], 'dates', f'risk of broker {b.id}', f'broker risk time series {b.id}', f'riskseries{b.id}.png')

    random_ids = random.sample(brokers, 5)
    value_series = [broker_statuses[b.id] for b in random_ids]
    risk_series = [broker_risks[b.id] for b in random_ids]
    plot_times(dates, value_series, [f'Broker {i.id}' for i in random_ids], 'Dates', 'Broker wealth',
               'Broker wealth time series', f'timeseriesJoint.png')
    plot_times(dates, risk_series, [f'Broker {i.id}' for i in random_ids], 'Dates', 'Broker wealth',
               'Broker risk time series', f'riskseriesJoint.png')

    # plot portfolio risk to portfolio value
    final_portfolio_values = [broker_statuses[b.id][-1] for b in brokers]

    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    plt.hist([math.floor(x) for x in final_portfolio_values], bins=50, color='skyblue', alpha=.5, density=True)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Final Portfolio Values')
    plt.savefig(f'figures/finalValueHistogram')
    plt.show()

    portfolio_risk = [broker.current_risk for broker in brokers]
    plot(portfolio_risk, final_portfolio_values, 'Portfolio risk', 'Final portfolio value',
         'Final portfolio values to risk', 'riskToValue.png')

    # plot influence in to portfolio value
    influences = [sum(broker_network.get_edge_data(n, me)['weight'] for (n, me) in broker_network.in_edges(broker.id)) for broker in brokers]
    plot(influences, final_portfolio_values, 'Influence summed inputs', 'Final portfolio value',
         'Portfolio value to influence', 'valueToInfluence.png')

    # plot number of friends to portfolio value
    num_friends = [len(broker_network.in_edges(broker.id)) for broker in brokers]
    plot(num_friends, final_portfolio_values, 'Number of friends', 'Final portfolio value',
         'Final portfolio value to number of friends', 'valueToFriendCount.png')


    # plot values mid interval
    interim_portfolio_values = [broker_statuses[b.id][N//2] for b in brokers]
    portfolio_risk = [broker.current_risk for broker in brokers]
    plot(portfolio_risk, interim_portfolio_values, 'Portfolio risk', 'Final portfolio value',
         'Interim portfolio values to risk', 'interimRiskToValue.png')

    # plot influence in to portfolio value
    influences = [sum(broker_network.get_edge_data(n, me)['weight'] for (n, me) in broker_network.in_edges(broker.id)) for broker in brokers]
    plot(influences, interim_portfolio_values, 'Influence summed inputs', 'Final portfolio value',
         'Interim portfolio value to influence', 'interimValueToInfluence.png')

    # plot number of friends to portfolio value
    num_friends = [len(broker_network.in_edges(broker.id)) for broker in brokers]
    plot(num_friends, interim_portfolio_values, 'Number of friends', 'Final portfolio value',
         'Interim portfolio value to number of friends', 'interimValueToFriendCount.png')
