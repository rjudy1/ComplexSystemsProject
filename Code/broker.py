import ast
import collections
from datetime import datetime, timedelta
import math
import random


# idea is to increase threshold for a given stock if neighbor reports a good value
# also need to define randomized order/buy amount of some kind
# need to figure out if risk reward balance plays in
class Broker:
    def __init__(self, id: int, initial_money: float, risk_minimum: float = 0, risk_maximum: float = math.inf,
                 purchase_rate_limit: int = 0, neighbor_weight: float = 0, shift_influence: bool = False):
        # let's say minimum risk is the minimum index out of 100 stocks to consider buying/selling when trying to raise or lower risk
        # one problem is that we're essentially defining low risk as diversification of portfolio
        self.id = id
        self.money = initial_money
        self._portfolio = collections.defaultdict(lambda: 0)  # stock to quantity dictionary

        self.neighbor_weight = neighbor_weight

        self.preferred_risk_minimum = risk_minimum
        self.preferred_risk_maximum = risk_maximum

        # TODO: set up preferred risk minimum to a percentage to consider of the available stocks (considers x% highest risk stocks)
        # TODO: establish a range of minimum risk preferences and map that minimum risk percentage to a value from 0 to 99 for x (see above)
        self.risk_percentile = math.exp(.08*self.preferred_risk_minimum) - 1
        if self.risk_percentile >= 1.0:
            self.risk_percentile = .99
        print(f'riskmin {risk_minimum}, percentile {self.risk_percentile}')

        self.current_risk = 0

        self.adjust_influence = shift_influence
        self.neighbor_history = dict()  # how do we want to use this?

        # possible configuration values for later
        self._purchase_rate = purchase_rate_limit

    # buy and sell functionality based on allowed purchase rate, outside influence,
    def update(self, graph, available_stocks, id_to_broker_map, stocks, date):
        # include transaction, update status
        self.assess_portfolio_risk(stocks, date)

        while (self.current_risk < self.preferred_risk_minimum or self.current_risk > self.preferred_risk_maximum or random.random() < .05) and self.money > 0:
            # is there a better way to choose then randomly selecting higher or lower risk
            # if large gap, pull from back, if small from front half?
            if self.preferred_risk_minimum > self.current_risk:
                # assess risk of every stock available and sort from lowest to highest risk
                risk_dict = dict()
                for stock in available_stocks:
                    risk_dict[stock] = self.assess_risk(stock, stocks, date) \
                                       + self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)[stock]
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)

                # select from stocks whose stock assessment put them in the risk min / 100 top percent of stocks
                lower_limit = math.floor(len(sorted_list) * self.risk_percentile)
                plan = random.sample(list(sorted_list[lower_limit:]), 1)[0]

                # buy stock and increment quantity in portfolio
                self.money -= stocks[plan].date_to_price[date]
                self._portfolio[plan] += 1

            elif self.preferred_risk_maximum < self.current_risk or self.money < 1000 or random.random() < .05:  # if above preferred risk or running out of money or noise, sell
                risk_dict = dict()
                for stock in self._portfolio:
                    risk_dict[stock] = (self.assess_risk(stock, stocks, date)
                                        + self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)[stock])

                if len(risk_dict) == 0:
                    break  # nothing can be done to lower the risk if they don't own any stocks
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)
                lower_limit = math.floor(len(sorted_list) * self.risk_percentile)
                plan = random.sample(list(sorted_list[lower_limit:]), 1)[0]

                self.money += stocks[plan].date_to_price[date]
                self._portfolio[plan] -= 1

            self.assess_portfolio_risk(stocks, date), self.id
            print(self.id, self.current_risk)

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
            for (n, s) in neighbors:
                stock_dict[stock] += graph.get_edge_data(n,self.id)['weight'] * id_to_broker_map[n].assess_risk(stock, stocks, date)
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
        current_amount = self._portfolio[ticker] * stocks[ticker].date_to_price[date]
        portfolio_allocation = (current_amount + stocks[ticker].date_to_price[date]) / (
                    self.get_status(stocks, date)
                    - self.money + stocks[ticker].date_to_price[date])
        portfolio_volume = self._portfolio[ticker]

        start = datetime.strptime(date, "%m/%d/%Y")
        high = 0
        low = math.inf
        for d in range(52*7):
            high = max(high, stocks[ticker].date_to_price[(start + timedelta(days=d)).strftime("%m/%d/%Y")])
            low = min(low, stocks[ticker].date_to_price[(start + timedelta(days=d)).strftime("%m/%d/%Y")])

        average = 0
        for d in range(52):
            average += stocks[ticker].date_to_price[(start + timedelta(days=d)).strftime("%m/%d/%Y")]
        average /= 52

        return portfolio_allocation, portfolio_volume, high, low, average

    def assess_risk(self, ticker, stocks, date):
        """
        Assess the risk using a fat-tailed distribution formula.
        @param ticker: ticker for stock to assess
        @param stocks: dataframe of stock data
        @param date: date in format MM/DD/YYYY

        Returns:
            float: Risk assessment for a single stock.
        """
        portfolio_allocation, _, high, low, fifty_day_average = self.get_stats_for_risk(ticker, stocks, date)

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
        portfolio_allocation, volumes, forward_eps_list, earnings_growth_list, high_list, low_list, dividend_ratio_list, fifty_day_average_list = list(), list(), list(), list(), list(), list(), list(), list()

        for ticker in self._portfolio:
            pa, volume, h, l, fda = self.get_stats_for_risk(ticker, stocks, date)
            portfolio_allocation.append(pa)
            volumes.append(volume)
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
            total_risk += individual_risk * volumes[i]**2

        self.current_risk = total_risk
        return total_risk
