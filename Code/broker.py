import collections
import math
from collections import defaultdict


# idea is to increase threshold for a given stock if neighbor reports a good value
# also need to define randomized order/buy amount of some kind
# need to figure out if risk reward balance plays in
class Broker:
    def __init__(self, id: int, initial_money: float, risk_minimum: float=0, risk_maximum: float=math.inf,
                 purchase_rate_limit: int=0, threshold: int=0, neighbor_influence: dict=dict()):
        self.id = id
        self.money = initial_money
        self._public_status = dict()  # contains percent improvements in last period for owned stocks?
        self._purchase_rate = purchase_rate_limit
        self._stocks = defaultdict(lambda: {})  # should we limit how many of each stock is avaiable?
        self._neighbors = neighbor_influence # not sure this should be inside the broker, maybe need to pass in when buying/selling a
                                             # and maintain the global graph
        self._threshold = threshold  # should threshold be determined by risk minimum and maximum?


    # buy and sell functionality based on allowed purchase rate, outside influence,
    def update(self, influence, stock, price, graph, available_stocks, history):
        # include transaction, update status
        # change weight of incoming edges based on values?
        # update public status
        for stock in available_stocks:
            # if combo of neighbor factor and stock change in given period, still need to check interval if using that
            if (history[stock][0] - history[stock][1]) and self.get_neighbor_factor(graph)[stock] > self._threshold: # types and indices are not right rn, maybe need weights
                # BUY
                # if chose to buy based on one neighbor, increase weight of those edges??
                # pick neighbor that had the greatest influence for this purchase increase by delta?
                pass
            elif (history[stock][0] - history[stock][1]) and self.get_neighbor_factor(graph)[stock] < self._threshold: # types and indices are not right rn, maybe need weights, change threshold to address risk
                # SELL
                # if chose to sell based on one neighbor, increase weight of those edges??
                pass
        # update public status

    def get_status(self):
        return self._public_status

    def get_neighbor_factor(self, graph, available_stocks, id_to_broker_map):  # need to clean up parameter choices
        neighbors = graph.in_edges(self.id)
        stock_dict = collections.defaultdict(lambda: 0)
        # iterate through graph getting neighbors and values - simple weighting?
        for stock in available_stocks:
            for n in neighbors:
                stock_dict[stock] += graph.weights[n][self.id] * id_to_broker_map[n].get_status(stock)

        return stock_dict


    # update function
