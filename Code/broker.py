import collections
import math
import random


# idea is to increase threshold for a given stock if neighbor reports a good value
# also need to define randomized order/buy amount of some kind
# need to figure out if risk reward balance plays in
class Broker:
    def __init__(self, id: int, initial_money: float, risk_minimum: float=0, risk_maximum: float=math.inf,
                 purchase_rate_limit: int=0, neighbor_weight: float=0, shift_influence = False):
        self.id = id
        self.money = initial_money
        self._portfolio = dict()  # contains percent improvements in last period for owned stocks?

        self.neighbor_weight = neighbor_weight

        self.preferred_risk_minimum = risk_minimum
        self.preferred_risk_maximum = risk_maximum
        self.current_risk = 0

        self.adjust_influence = shift_influence
        self.neighbor_history = dict()

        # possible configuration values for later
        self._purchase_rate = purchase_rate_limit

    # buy and sell functionality based on allowed purchase rate, outside influence,
    def update(self, graph, available_stocks, id_to_broker_map):
        # include transaction, update status
        self.update_current_risk(available_stocks)

        # assess risk of buying/selling each stock
        risk_dict = dict()
        for stock in available_stocks:
            risk_dict[stock] = self.get_risk(stock, available_stocks) + self.neighbor_weight * self.get_neighbor_factor(graph, available_stocks, id_to_broker_map)

        while self.current_risk < self.preferred_risk_minimum or self.current_risk > self.preferred_risk_maximum or random.random() < .05:
            # is there a better way to choose then randomly selecting higher or lower risk



            # if large gap, pull from back, if small from front half?
            if self.preferred_risk_minimum > self.current_risk:
                risk_dict = dict()
                for stock in available_stocks:
                    risk_dict[stock] = self.get_risk(stock, available_stocks) + self.get_neighbor_factor(graph,
                                                                                                         available_stocks,
                                                                                                         id_to_broker_map)

                sorted_list = sorted(risk_dict.keys(), key=lambda x: x[1])
                if self.preferred_risk_minimum - self.current_risk > .5 * self.preferred_risk_minimum:  # tiny bit arbitrary
                    # purchase random sample from upper half
                    plan = random.sample(list(sorted_list[len(sorted_list)//2:]))
                else:
                    plan = random.sample(list(sorted_list[:len(sorted_list)//2]))
                self.money -= available_stocks[plan].price
                self._portfolio[plan] += 1

            elif self.preferred_risk_maximum < self.current_risk:
                risk_dict = dict()
                for stock in self._portfolio:
                    risk_dict[stock] = self.get_risk(stock, available_stocks) + self.get_neighbor_factor(graph,
                                                                                                         available_stocks,
                                                                                                         id_to_broker_map)

                sorted_list = sorted(risk_dict.keys(), key=lambda x: x[1])
                if self.preferred_risk_minimum - self.current_risk > .5 * self.preferred_risk_minimum:  # tiny bit arbitrary
                    # purchase random sample from upper half
                    plan = random.sample(list(sorted_list[len(sorted_list)//2:]))
                else:
                    plan = random.sample(list(sorted_list[:len(sorted_list)//2]))
                self.money += available_stocks[plan].price
                self._portfolio[plan] -= 1

            self.update_current_risk(available_stocks)

        # update public status
        if self.adjust_influence:
            for neighbor in graph.in_edges(self.id):
                if id_to_broker_map[neighbor].get_status(available_stocks) > self.neighbor_history[neighbor]:
                    graph.weights[neighbor][self.id] += .005  # TODO: Parameterize this
                if id_to_broker_map[neighbor].get_status(available_stocks) < self.neighbor_history[neighbor]:
                    graph.weights[neighbor][self.id] += .005  # TODO: Parameterize this

                self.neighbor_history[neighbor] = id_to_broker_map[neighbor].get_status(available_stocks)

    def get_status(self, stock_map):
        return sum(self._portfolio[ticker]*stock_map[ticker] for ticker in self._portfolio) + self.money

    def get_neighbor_factor(self, graph, available_stocks, id_to_broker_map):  # need to clean up parameter choices
        neighbors = graph.in_edges(self.id)
        stock_dict = collections.defaultdict(lambda: 0)
        # iterate through graph getting neighbors and values - simple weighting?
        for stock in available_stocks:
            for n in neighbors:
                stock_dict[stock] += graph.weights[n][self.id] * id_to_broker_map[n].get_risk(stock, available_stocks)
        return stock_dict

    def update_current_risk(self, available_stocks):
        '''
        Update the current risk based on the updated data

        :param available_stocks:

        :return: None
        '''
        # TODO: this will update based on new data
        pass

    def get_risk(self, ticker, available_stocks, include_portfolio=False) -> float:
        '''

        :param ticker: stock to assess
        :param available_stocks: map of tickers to stock data, possible Stock class
        :param include_portfolio: determines whether or not to include own portfolio when assessing risk

        :return: normalized float with risk for a given stock
        '''

        # TODO: populate this with computation for risk given stats - need something that says risk of buying or selling
        # @Josh is this calculation going to account for history? might need another parameter in Stock for that
        pass

