import math
import matplotlib.pyplot as plt
import networkx as nx
import random

from broker import Broker

# will have directed graph indicating how much influence
broker_network = nx.DiGraph()

# we need to pull in data over relevant intervals - maybe
# create list of brokers
N = 100
brokers = [Broker(i, 100) for i in range(N)]

broker_network.add_nodes_from([node.id for node in brokers])

# need three simulations? different days/times frequency allowed to trade,

# add edges somehow setting influence with power law distribution? - need to better define influence though
# How should the influence be adjusted?
broker_network.add_edges_from([(2, 3, {'weight': 8})])

# change with broker_network[source][dest][weight]

nx.draw(broker_network, with_labels = True, font_color = 'white', node_shape = 's')
plt.show()