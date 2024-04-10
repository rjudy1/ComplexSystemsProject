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
filename = 'data/ticker_info_random1000.csv'
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
    if math.isnan(earnings_growth):
        earnings_growth = .05
    if math.isnan(forward_eps):
        forward_eps = 2

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
end = "12/31/2010"

# test over 1980 to 2000 and then 2003 to 2023

# figure out what data to collect and plot from the simulation

for trial in range(trials):
    # stats per trial
    broker_statuses = defaultdict(lambda: list())
    dates = list()  # for labels

    # brokers will need to set these values based on if we're doing a influence shift or a risk variety

    brokers = [Broker(i, 1_000_000, i/N*15, 15, 0, 0.05, True)
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
            if broker_statuses[broker.id][-1] < 0:
                print(broker.money, broker.portfolio)
            # print(broker.id, broker.get_status(stocks, date.strftime("%m/%d/%Y")))

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

    def plot(x, y, xlabel, ylabel, title, filename):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o')
        # Customize the plot
        plt.title(title)
        # tick_locations = [len(x)//7*i for i in range(8)]
        # labels = [x[t] for t in tick_locations[:-1]]
        # labels.append(x[-1])
        # plt.xticks(tick_locations, labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
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

    plot(broker_ids, num_stocks, 'broker ids', 'num stocks', 'num stocks owned', 'numstocks.png')
    plot(broker_ids, num_unique_stocks, 'broker ids', 'num stocks', 'num unique stocks owned', 'numuniquestocks.png')
    plot(broker_ids, cash_at_end, 'broker ids', 'money', 'money at end', 'numstocks.png')


    # dates = ['1/1/2003', '5/1/2003', '9/1/2003', '1/1/2004']
    # plot some time series of brokers net worth
    for b in random.sample(brokers, 10):
        plot_time(dates, broker_statuses[b.id], 'dates', f'wealth of broker {b.id}', f'broker wealth time series {b.id}', f'timeseries{b.id}.png')

    # plot portfolio risk to portfolio value
    final_portfolio_values = [broker_statuses[b.id][-1] for broker in brokers]

    # bar plot
    # Creating indices for each value
    indices = range(len([math.floor(x) for x in final_portfolio_values]))

    # Plotting the bar graph
    plt.figure(figsize=(8, 6))
    plt.bar(indices, [math.floor(x) for x in final_portfolio_values], color='skyblue')

    # Adding labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Plot of Values')

    # Display the plot
    plt.show()


    portfolio_risk = [broker.current_risk for broker in brokers]
    plot(portfolio_risk, final_portfolio_values, 'Portfolio risk', 'Final portfolio value', 'Final Portofolio values', 'risktovalue.png')

    # plot influence in to portfolio value
    influences = [sum(broker_network.get_edge_data(n, me)['weight'] for (n, me) in broker_network.in_edges(broker.id)) for broker in brokers]
    plot(influences, final_portfolio_values, 'Influence inputs', 'Final portfolio value', 'influence to value', 'influenceintovalue.png')

    # plot number of friends to portfolio value
    num_friends = [len(broker_network.in_edges(broker.id)) for broker in brokers]
    plot(num_friends, final_portfolio_values, 'Number of friends', 'Final portfolio value', 'influence to value', 'influenceintovalue.png')


    # plot values pre 2008
    final_portfolio_values = [broker_statuses[b.id][N//2] for broker in brokers]
    portfolio_risk = [broker.current_risk for broker in brokers]
    plot(portfolio_risk, final_portfolio_values, 'Portfolio risk', 'Final portfolio value', 'Final Portofolio values', 'risktovalue.png')

    # plot influence in to portfolio value
    influences = [sum(broker_network.get_edge_data(n, me)['weight'] for (n, me) in broker_network.in_edges(broker.id)) for broker in brokers]
    plot(influences, final_portfolio_values, 'Influence inputs', 'Final portfolio value', 'influence to value', 'influenceintovalue.png')

    # plot number of friends to portfolio value
    num_friends = [len(broker_network.in_edges(broker.id)) for broker in brokers]
    plot(num_friends, final_portfolio_values, 'Number of friends', 'Final portfolio value', 'influence to value', 'influenceintovalue.png')
