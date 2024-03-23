import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import powerlaw
import random

from broker import Broker

# will have directed graph indicating how much influence
broker_network = nx.DiGraph()

# we need to pull in data over relevant intervals - maybe
# create list of brokers
N = 100
standard_influence = .2
time_steps = 1000
epochs = 100

# figure out what data to collect and plot from the simulation

for epoch in range(epochs):
    # (self, id: int, initial_money: float, risk_minimum: float=0, risk_maximum: float=math.inf,
    #                  purchase_rate_limit: int=0, neighbor_weight: float=0, shift_influence = False)
    # brokers will need to set these values based on if we're doing a influence shift or a risk variety

    brokers = [Broker(i,  100) for i in range(N)]

    # Generate some synthetic data following a power law distribution
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


    nx.draw(broker_network, with_labels=True, font_color='white', node_shape='s')
    plt.show()

    for step in range(time_steps):
        random.shuffle(brokers)
        # TODO: replace with data for the time step, possible turn into map of stock to price/quantity available
        available_stocks = {}

        for broker in brokers:
            broker.update(broker_network, available_stocks)

    # examine state of brokers at end

    for broker in brokers:
        print(broker.get_status())

# add plotting code from whatever we decide to plot

