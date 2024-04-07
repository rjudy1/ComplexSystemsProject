import ast
import collections
import math
import random


# idea is to increase threshold for a given stock if neighbor reports a good value
# also need to define randomized order/buy amount of some kind
# need to figure out if risk reward balance plays in
class Broker:
    def __init__(self, id: int, initial_money: float, risk_minimum: float = 0, risk_maximum: float = math.inf,
                 purchase_rate_limit: int = 0, neighbor_weight: float = 0, shift_influence: bool = False):
        self.id = id
        self.money = initial_money
        self._portfolio = collections.defaultdict(lambda: 0)  # stock to quantity dictionary

        self.neighbor_weight = neighbor_weight

        self.preferred_risk_minimum = risk_minimum
        self.preferred_risk_maximum = risk_maximum
        self.current_risk = 0

        self.adjust_influence = shift_influence
        self.neighbor_history = dict()  # how do we want to use this?

        # possible configuration values for later
        self._purchase_rate = purchase_rate_limit

    # buy and sell functionality based on allowed purchase rate, outside influence,
    def update(self, graph, available_stocks, id_to_broker_map, stocks, date):
        # include transaction, update status
        self.assess_portfolio_risk(stocks, date)

        # assess risk of buying/selling each stock
        risk_dict = dict()
        for stock in available_stocks:
            risk_dict[stock] = (self.assess_risk(stock, stocks, date) + self.neighbor_weight *
                                self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)[stock])
        print("finished individiual assessments")
        while self.current_risk < self.preferred_risk_minimum or self.current_risk > self.preferred_risk_maximum or random.random() < .05:
            # is there a better way to choose then randomly selecting higher or lower risk
            print(self.current_risk)
            # if large gap, pull from back, if small from front half?
            if self.preferred_risk_minimum > self.current_risk:
                risk_dict = dict()
                for stock in available_stocks:
                    risk_dict[stock] = self.assess_risk(stock, stocks, date) \
                                       + self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)[stock]
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)
                if self.preferred_risk_minimum - self.current_risk > .5 * self.preferred_risk_minimum:  # tiny bit arbitrary
                    # purchase random sample from upper half
                    plan = random.sample(list(sorted_list[len(sorted_list) // 2:]), 1)[0]
                else:
                    plan = random.sample(list(sorted_list[:len(sorted_list) // 2]), 1)[0]
                self.money -= stocks[plan].date_to_price[date]
                self._portfolio[plan] += 1

            elif self.preferred_risk_maximum < self.current_risk:
                risk_dict = dict()
                for stock in self._portfolio:
                    risk_dict[stock] = (self.assess_risk(stock, stocks, date)
                                        + self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)[stock])

                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)
                if self.preferred_risk_minimum - self.current_risk > .5 * self.preferred_risk_minimum:  # tiny bit arbitrary
                    # purchase random sample from upper half
                    plan = random.sample(list(sorted_list[len(sorted_list) // 2:]), 1)[0]
                else:
                    plan = random.sample(list(sorted_list[:len(sorted_list) // 2]), 1)[0]
                self.money += stocks[plan].date_to_price[date]
                self._portfolio[plan] -= 1

            self.assess_portfolio_risk(stocks, date)

        # update public status
        if self.adjust_influence:
            for neighbor in graph.in_edges(self.id):
                graph.weights[neighbor][self.id] += .005 * (  # TODO: Parameterize this
                        id_to_broker_map[neighbor].get_status(available_stocks) - self.neighbor_history[neighbor])
                self.neighbor_history[neighbor] = id_to_broker_map[neighbor].get_status(available_stocks)

    def get_status(self, stocks, date):
        """
        Calculates the value of the broker's stock portfolio given the status of the stocks on a given date assuming the
            broker were to sell immediately
        :param stocks: the stock pandas dataframe
        :param date: date in format 'MM/DD/YYYY'
        :return: float of current value of broker
        """

        portfolio_value = self.money

        for ticker in self._portfolio:
            portfolio_value += self._portfolio[ticker] * stocks[ticker].date_to_price[date]

        return portfolio_value

    def get_neighbor_factor(self, graph, available_stocks, id_to_broker_map, stocks, date):
        neighbors = graph.in_edges(self.id)
        stock_dict = collections.defaultdict(lambda: 0)
        # iterate through graph getting neighbors and values - simple weighting?
        for stock in available_stocks:
            for n in neighbors:
                stock_dict[stock] += graph.weights[n][self.id] * id_to_broker_map[n].assess_risk(stock, stocks, date)
        return stock_dict  # dictionary of neighbor risk assessment of each stock in the available stocks

    def get_stats_for_risk(self, ticker, stocks, date):
        """
        Computes the stats for Josh's risk functions
        @param ticker: ticker for stock to assess
        @param stocks: dataframe of stock data
        @param date: date in format MM/DD/YYYY
        :return: tuple of portfolio_allocation, forward_eps, earnings_growth, high, low, dividend_ratio, fifty_day_average
        """

        # compute total equity for a given stock
        index = stocks[ticker].dates.index(date)

        current_amount = self._portfolio[ticker] * stocks[ticker].date_to_price[date]
        portfolio_allocation = (current_amount + stocks[ticker].date_to_price[date]) / (
                    self.get_status(stocks, date)
                    - self.money + stocks[ticker].date_to_price[date])

        start = index - 52*7+1 if index >= 52*7-1 else 0
        high = 0
        low = math.inf
        for d in range(start, index+1):
            high = max(high, stocks[ticker].date_to_price[stocks[ticker].dates[d]])
            low = min(low, stocks[ticker].date_to_price[stocks[ticker].dates[d]])

        start = index - 49 if index >= 49 else 0
        average = 0
        for d in range(start, index+1):
            average += stocks[ticker].date_to_price[stocks[ticker].dates[d]]
        average /= 52

        return portfolio_allocation, high, low, average

    def assess_risk(self, ticker, stocks, date):
        """
        Assess the risk using a fat-tailed distribution formula.
        @param ticker: ticker for stock to assess
        @param stocks: dataframe of stock data
        @param date: date in format MM/DD/YYYY

        Returns:
            float: Risk assessment for a single stock.
        """
        portfolio_allocation, high, low, fifty_day_average = self.get_stats_for_risk(ticker, stocks, date)

        k = 3  # Constant for fat-tailed distribution

        x = math.exp(abs(high - low - stocks[ticker].dividend_ratio * fifty_day_average) * -k)  # Equation to express volatility
        a = portfolio_allocation * stocks[ticker].forward_eps * (
                    1 - stocks[ticker].earnings_growth)  # Equation to express likelihood (a weighted sum of significant metrics)

        return a * (x) if not math.isnan(a*x) else 0

    def assess_portfolio_risk(self, stocks, date):
        """
        Assess the risk of the entire portfolio using a fat-tailed distribution formula.

        @param stocks: dataframe of stock data
        @param date: date in format MM/DD/YYYY

        Returns:
            float: Risk assessment for the entire portfolio.

        """
        portfolio_allocation, forward_eps_list, earnings_growth_list, high_list, low_list, dividend_ratio_list, fifty_day_average_list = list(), list(), list(), list(), list(), list(), list()

        for ticker in self._portfolio:
            pa, h, l, fda = self.get_stats_for_risk(ticker, stocks, date)
            portfolio_allocation.append(pa)
            forward_eps_list.append(stocks[ticker].forward_eps)
            earnings_growth_list.append(stocks[ticker].earnings_growth)
            high_list.append(h)
            low_list.append(l)
            dividend_ratio_list.append(stocks[ticker].dividend_ratio)
            fifty_day_average_list.append(fda)

        k = 3  # Constant for fat-tailed distribution
        total_risk = 0
        for i in range(len(portfolio_allocation)):
            x = math.exp(abs(high_list[i] - low_list[i] - dividend_ratio_list[i] * fifty_day_average_list[i]) * -k)  # volatility
            a = portfolio_allocation[i] * forward_eps_list[i] * (
                    1 - earnings_growth_list[i])  # likelihood
            individual_risk = a * x
            total_risk += individual_risk

        self.current_risk = total_risk
        return total_risk
