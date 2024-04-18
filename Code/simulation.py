import ast
from datetime import datetime, timedelta
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import powerlaw
import random
from time import time

from broker import Broker
from stock import Stock


# -------------------------------------------------------------------------------------------------------------------- #
# Parameters for simulation - using 10000 as the normalizing value for risk percentile

N = 100  # number of brokers
neighbor_influence = 0.04  # initial percent of risk assessment provided by each neighbor
adjust_influence = True
use_influence_only = True
seed_money = 1_000_000
stop_at_stable = True
trials = 1  # not really using this since just doing a single run currently since its such a long runtime
start = "01/01/2003"  # date to start simulation
end = "12/31/2012"  # date to stop simulation
input_data_filename = 'data/ticker_info_400_all_catagories.csv'
figure_file_directory = 'figures_influenceonly2003-2012_ni4_stop-at-stable'
# -------------------------------------------------------------------------------------------------------------------- #


# helper functions
def plot_times(x, ys, serieslabels, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for i in range(len(ys)):
        plt.plot(x, ys[i], label=f'{serieslabels[i]}', linewidth=.5)

    tick_locations = [len(x) // 7 * i for i in range(8)]
    labels = [x[t] for t in tick_locations[:-1]]
    labels.append(x[-1])
    plt.xticks(tick_locations, labels)

    # Customize the plot
    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel), plt.grid(True)
    plt.legend()
    plt.savefig(f'{figure_file_directory}/{filename}')
    plt.show()


def plot(x, y, xlabel, ylabel, title, filename):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o')
    plt.xlabel(xlabel), plt.ylabel(ylabel), plt.title(title)
    plt.grid(True)
    plt.savefig(f'{figure_file_directory}/{filename}')
    plt.show()

# -------------------------------------------------------------------------------------------------------
# create directory if it does not exist 
if not os.path.isdir(figure_file_directory):
    os.makedirs(figure_file_directory)

# -------------------------------------------------------------------------------------------------------
# pull data from csv file of stock data
print(f"Reading pandas dataframe of stocks at {input_data_filename} and converting to dictionary of Stock at {time()}")
stock_df = pd.read_csv(input_data_filename)
valid_tickers = list(stock_df['Ticker'])  # allow indexing of tickers
stock_df.set_index('Ticker', inplace=True)

# convert to dictionary of ticker to Stock class objects
stocks = dict()
random.shuffle(valid_tickers)
for idx, ticker in enumerate(valid_tickers):
    try:
        dates = ast.literal_eval(stock_df.at[ticker, 'dates'])
        prices = ast.literal_eval(stock_df.at[ticker, 'time_series'])
    except ValueError:  # skip the malformatted date/price strings
        continue

    forward_eps = stock_df['forwardEps'][ticker]
    earnings_growth = stock_df['earningsGrowth'][ticker]
    dividend_ratio = stock_df['dividendRate'][ticker]

    stocks[ticker] = Stock(ticker, forward_eps, earnings_growth, dividend_ratio, dates, prices)
    if idx % 100 == 0:
        print(f"completed ticker {ticker} at index {idx} at time {time()}...")
    if idx > 175:  # TODO: Check if this restriction affects portfolio size
        break  # cut off sooner to speed up integration testing

print(f"stocks dictionary created successfully ({time()})...")


# -------------------------------------------------------------------------------------------------------
# run simulation
for trial in range(trials):
    # stats per timestep - broker portfolio status, current risk status
    dates = list()  # for labels
    broker_statuses = defaultdict(lambda: list())
    broker_risks = defaultdict(lambda: list())
    money_series = defaultdict(lambda: list())

    # list of brokers with increasing preferred levels of risk
    brokers = [Broker(i, seed_money, (i+1)/N*9999, neighbor_influence, adjust_influence, stop_at_stable=stop_at_stable) for i in range(N)]
    # version with limited risk types to focus on influence
    if use_influence_only:
        def choose_risk(id):
            if id < 33:
                return 300
            elif id < 66:
                return 5000
            else:
                return 9999
        brokers = [Broker(i, seed_money, choose_risk(i), neighbor_influence, adjust_influence, stop_at_stable=stop_at_stable) for i in range(N)]

    # @INFLUENCE THIS BLOCK ADDRESSES THE NEIGHBOR WEIGHTING AND SETUP
    # Generate the number of friends per person with power distribution and populate those friend relationships
    # with a normal distribution of the available brokers in both the Broker class and the network visualization
    friend_count_dist = np.floor(np.array(powerlaw.Power_Law(xmin=2, parameters=[2.7]).generate_random(N)))
    if use_influence_only:
        friend_count_dist.sort()
        # sort distribution
        list1 = [friend_count_dist[i*3] for i in range(len(friend_count_dist)//3)]
        list2 = [friend_count_dist[i*3+1] for i in range(len(friend_count_dist)//3)]
        list3 = [friend_count_dist[i*3+2] for i in range(len(friend_count_dist)//3)]
        friend_count_dist = [*list1, *list2, *list3, friend_count_dist[len(friend_count_dist)-1]]
    broker_network = nx.DiGraph()
    broker_network.add_nodes_from([node.id for node in brokers])
    for i in range(N):
        for j in range(int(friend_count_dist[i])):
            neighbor = abs(math.ceil(np.random.uniform(0,N-1))) #(N/2, .34*N))) % 100
            if neighbor != i:
                broker_network.add_edge(i, neighbor, weight=1/friend_count_dist[i])
                brokers[neighbor].in_neighbors[i] = 1 / friend_count_dist[i]  # not a clean way to do this but it'll work hopefully

    for b in brokers:
        b.neighbor_weight = neighbor_influence * len(b.in_neighbors)
        for neighbor in b.in_neighbors:
            b.in_neighbors[neighbor] = neighbor_influence

    # display the network created
    # nx.draw(broker_network, with_labels=True, font_color='white', node_shape='s')
    # plt.show()

    start_date, end_date = datetime.strptime(start, "%m/%d/%Y"),  datetime.strptime(end, "%m/%d/%Y")

    available_stocks = set()

    # iterate through all stocks in dataframe and add ones that exist at this time to the available set
    for ticker in stocks:
        stock = stocks[ticker]
        if start_date.strftime("%m/%d/%Y") in stock.date_to_price and stock.date_to_price[
            start_date.strftime("%m/%d/%Y")] > 0.00001:
            available_stocks.add(ticker)

    if len(available_stocks):
        # update
        # populate the initial portfolio uniformly? or do the risk assessment and populate with riskiest stocks?
        # burn .5% of portfolio per stock
        for b in brokers:
            initial_unique_stocks = 200
            for i in range(initial_unique_stocks):
                ticker = random.sample(list(available_stocks), 1)[0]
                quantity = max(1, seed_money / initial_unique_stocks // stocks[ticker].date_to_price[start_date.strftime("%m/%d/%Y")])
                b.portfolio[ticker] += quantity
                b.money -= quantity * stocks[ticker].date_to_price[start_date.strftime("%m/%d/%Y")]

    date = start_date
    curr_time = time()
    broker_ids = [b.id for b in brokers]
    while date < end_date:
        print(date.strftime("%m/%d/%Y"), f'-------{time()-curr_time}-------------------------------------')
        curr_time = time()
        dates.append(date.strftime('%m/%d/%Y'))

        random.shuffle(broker_ids)
        available_stocks = set()

        # iterate through all stocks in dataframe and add ones that exist at this time to the available set
        for ticker in stocks:
            stock = stocks[ticker]
            if date.strftime("%m/%d/%Y") in stock.date_to_price and stock.date_to_price[date.strftime("%m/%d/%Y")] > 0.00001:
                available_stocks.add(ticker)

        if len(available_stocks):
            prev = time()
            for i, bid in enumerate(broker_ids):
                # broker.money += 1_000
                brokers[bid].update(broker_network, available_stocks, brokers, stocks, date.strftime("%m/%d/%Y"))
                prev = time()

        # add dividend
        if date.day == 1 and (date.month == 12):
            for broker in brokers:
                for ticker in stocks:
                    broker.money += stocks[ticker].dividend_ratio

        for broker in brokers:
            broker_statuses[broker.id].append(broker.get_status(stocks, date.strftime("%m/%d/%Y")))
            broker_risks[broker.id].append(broker.current_risk)
            money_series[broker.id].append(broker.money)
            if broker_statuses[broker.id][-1] < 0:
                print(broker.id, broker.get_status(stocks, date.strftime("%m/%d/%Y")), broker.money, broker.current_risk, broker.portfolio)

        date += timedelta(days=1)

    # fix all the broker network edges for the end of the
    for b in brokers:
        for n in b.in_neighbors:
            broker_network.edges[n, b.id]['weight'] = b.in_neighbors[n]


    # -------------------------------------------------------------------------------------------------------
    # Plotting code for the results of the simulation

    # draw final state of the influence network
    pos = nx.spring_layout(broker_network, seed=3)
    nx.draw_networkx(broker_network, pos)
    for edge in broker_network.edges(data='weight'):
        nx.draw_networkx_edges(broker_network, pos, edgelist=[edge], width=edge[2]*2)
    plt.savefig(f'{figure_file_directory}/finalNeighborWeightsGraph.png')
    plt.show()

    # compute average value of brokers
    total_wealth = sum(broker_statuses[b.id][-1] for b in brokers)
    print(f'beginning wealth: 1_000_000')
    print(f'average ending wealth: {total_wealth/N}')

    broker_ids, num_stocks, num_unique_stocks, cash_at_end, dollar_per_stock = list(), list(), list(), list(), list()
    for b in brokers:
        broker_ids.append(b.id)
        count, unique_count = 0, 0
        portfolio_quant = 0
        for p in b.portfolio:
            count += b.portfolio[p]
            # get average percent of portfolio per stock in portfolio
            portfolio_quant += b.portfolio[p]*stocks[p].date_to_price[end_date.strftime('%m/%d/%Y')]
            unique_count += 1 if b.portfolio[p] > 0 else 0
        # print(b.portfolio)
        num_stocks.append(count)
        dollar_per_stock.append(portfolio_quant/count)
        num_unique_stocks.append(unique_count)
        cash_at_end.append(b.money)

    # plot statistics of portfolio at end
    plot(broker_ids, num_stocks, 'Broker ids as ordered by risk level', 'number of stocks owned',
         f'Number of stocks owned to broker id/risk minimum', 'stocksOwnedToBrokerIds.png')
    plot(broker_ids, dollar_per_stock, 'Broker ids as ordered by risk level', 'average dollar per stock',
         f'Average dollar per stock owned to broker id', 'averageDollarPerStockToBrokerIds.png')
    plot(broker_ids, num_unique_stocks, 'Broker ids as ordered by risk level', 'number of unique stocks owned',
         f'Unique stock count to broker id', 'uniqueStockCountToBrokerIds.png')
    plot(broker_ids, cash_at_end, 'Broker ids as ordered by risk level', 'Liquid currency ($)',
         f'Liquid currency at end of simulation {end}', f'liquidCurrency.png')

    # # plot some time series of brokers wealth, risk, and liquid money, randomly select 10 per go
    # for b in random.sample(brokers, 1):
    #     plot_times(dates, [broker_statuses[b.id]], [f'broker {b.id}'], 'dates', f'portfolio and currency value of broker {b.id}', f'broker wealth time series {b.id}', f'timeseries{b.id}.png')
    #     plot_times(dates, [broker_risks[b.id]], [f'broker {b.id}'], 'dates', f'risk of broker {b.id}', f'broker risk time series {b.id}', f'riskseries{b.id}.png')

    if not use_influence_only:
        random_ids = sorted(random.sample(brokers, 10), key=lambda b: b.id)
        plot_times(dates, [broker_statuses[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker wealth', 'Broker wealth (including portfolio and cash) time series', f'timeSeriesJoint.png')
        plot_times(dates, [broker_risks[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker risk', 'Broker risk time series', f'riskSeriesJoint.png')
        plot_times(dates, [money_series[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker liquid money', 'Broker liquid money time series', f'moneySeriesJoint.png')

        random_ids = sorted(random.sample(brokers, 10), key=lambda b: b.id)
        plot_times(dates, [broker_statuses[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker wealth', 'Broker wealth (including portfolio and cash) time series', f'timeSeriesJoint2.png')
        plot_times(dates, [broker_risks[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker risk', 'Broker risk time series', f'riskSeriesJoint2.png')
        plot_times(dates, [money_series[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker liquid money', 'Broker liquid money time series', f'moneySeriesJoint2.png')

        random_ids = sorted(random.sample(brokers, 10), key=lambda b: b.id)
        plot_times(dates, [broker_statuses[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker wealth', 'Broker wealth (including portfolio and cash) time series', f'timeSeriesJoint3.png')
        plot_times(dates, [broker_risks[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker risk', 'Broker risk time series', f'riskSeriesJoint3.png')
        plot_times(dates, [money_series[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker liquid money', 'Broker liquid money time series', f'moneySeriesJoint3.png')

    else:
        # brokers.sort(key=lambda b:b.id)
        random_ids = sorted(random.sample(brokers[:len(brokers)//3], 10), key=lambda b: b.id)
        plot_times(dates, [broker_statuses[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker wealth', 'Broker wealth (including portfolio and cash) time series with low risk',
                   f'timeSeriesJoint2_setRiskLow.png')
        plot_times(dates, [broker_risks[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker risk', 'Broker risk time series with low risk', f'riskSeriesJoint2_setRiskLow.png')
        plot_times(dates, [money_series[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker liquid money', 'Broker liquid money time series with low risk', f'moneySeriesJoint2_setRiskLow.png')

        random_ids = sorted(random.sample(brokers[len(brokers)//3:len(brokers)//3*2], 10), key=lambda b: b.id)
        plot_times(dates, [broker_statuses[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker wealth', 'Broker wealth (including portfolio and cash) time series with medium risk',
                   f'timeSeriesJoint2_setRiskMed.png')
        plot_times(dates, [broker_risks[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker risk', 'Broker risk time series with medium risk', f'riskSeriesJoint2_setRiskMed.png')
        plot_times(dates, [money_series[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker liquid money', 'Broker liquid money time series with medium risk', f'moneySeriesJoint2_setRiskMed.png')

        random_ids = sorted(random.sample(brokers[len(brokers)//3*2:], 10), key=lambda b: b.id)
        plot_times(dates, [broker_statuses[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker wealth', 'Broker wealth (including portfolio and cash) time series with high risk',
                   f'timeSeriesJoint2_setRiskHigh.png')
        plot_times(dates, [broker_risks[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker risk', 'Broker risk time series with high risk', f'riskSeriesJoint2_setRiskHigh.png')
        plot_times(dates, [money_series[b.id] for b in random_ids], [f'Broker {i.id}' for i in random_ids],
                   'Dates', 'Broker liquid money', 'Broker liquid money time series with high risk', f'moneySeriesJoint2_setRiskHigh.png')

    # plot portfolio risk to portfolio value
    final_portfolio_values = [broker_statuses[b.id][-1] for b in brokers]

    # Plotting the histogram
    plt.figure(figsize=(8, 6))
    plt.hist([math.floor(x) for x in final_portfolio_values], bins=50, color='skyblue', alpha=.5, density=True)
    plt.xlabel('Percentage')
    plt.ylabel('Portfolio Value')
    plt.title('Final Portfolio Values')
    plt.savefig(f'{figure_file_directory}/finalValueHistogram')
    plt.show()

    # plot value to risk, influence, friends
    portfolio_risk = [broker.current_risk for broker in brokers]
    plot(portfolio_risk, final_portfolio_values, 'Portfolio risk', 'Final portfolio value',
         'Final portfolio values to risk', 'valueToRisk.png')

    influences = [broker.neighbor_weight for broker in brokers]
    plot(influences, final_portfolio_values, 'Influence summed inputs', 'Final portfolio value',
         'Portfolio value to influence', 'valueToInfluence.png')
    if use_influence_only:
        influences = [broker.neighbor_weight for broker in brokers[:len(brokers) // 3]]
        plot(influences, final_portfolio_values[:len(brokers)//3], 'Influence summed inputs', 'Final portfolio value',
             'Portfolio value to influence with low risk', 'valueToInfluenceLowRisk.png')
        influences = [broker.neighbor_weight for broker in brokers[len(brokers) // 3:len(brokers)//3*2]]
        plot(influences, final_portfolio_values[len(brokers) // 3:len(brokers)//3*2], 'Influence summed inputs', 'Final portfolio value',
             'Portfolio value to influence with medium risk', 'valueToInfluenceMedRisk.png')
        influences = [broker.neighbor_weight for broker in brokers[len(brokers)//3*2:]]
        plot(influences, final_portfolio_values[len(brokers)//3*2:], 'Influence summed inputs', 'Final portfolio value',
             'Portfolio value to influence with high risk', 'valueToInfluenceHighRisk.png')


    num_friends = [len(broker.in_neighbors) for broker in brokers]
    for b in random.sample(brokers, 10):
        print(b.id, b.in_neighbors, b.neighbor_weight)

    plot(num_friends, final_portfolio_values, 'Number of friends', 'Final portfolio value',
         'Final portfolio value to number of friends', 'valueToFriendCount.png')

    # plot values mid interval
    interim_portfolio_values = [broker_statuses[b.id][len(dates)//2] for b in brokers]

    # portfolio_risk = [broker.current_risk for broker in brokers]
    plot(portfolio_risk, interim_portfolio_values, 'Portfolio risk', 'Final portfolio value',
         'Interim portfolio values to risk', 'interimRiskToValue.png')

    # influences = [sum(broker_network.get_edge_data(n, me)['weight'] for (n, me) in broker_network.in_edges(broker.id)) for broker in brokers]
    plot(influences, interim_portfolio_values, 'Influence summed inputs', 'Final portfolio value',
         'Interim portfolio value to influence', 'interimValueToInfluence.png')

    # num_friends = [len(broker_network.in_edges(broker.id)) for broker in brokers]
    plot(num_friends, interim_portfolio_values, 'Number of friends', 'Final portfolio value',
         'Interim portfolio value to number of friends', 'interimValueToFriendCount.png')
