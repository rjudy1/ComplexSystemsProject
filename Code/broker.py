import ast
import collections
from datetime import datetime, timedelta
import math
import random
import numpy as np


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
        self.portfolio = collections.defaultdict(lambda: 0)  # stock to quantity dictionary

        self.in_neighbors = dict()
        self.neighbor_weight = neighbor_weight

        self.preferred_risk_minimum = risk_minimum
        self.preferred_risk_maximum = risk_maximum

        # TODO: set up preferred risk minimum to a percentage to consider of the available stocks (considers x% highest risk stocks)
        self.risk_percentile = math.exp(.05*self.preferred_risk_minimum) - 1
        if self.risk_percentile >= 1.0:
            self.risk_percentile = .99

        self.current_risk = 0
        # self.current_status = 0

        self.adjust_influence = shift_influence
        self._neighbor_history = collections.defaultdict(lambda: initial_money)  # how do we want to use this?

        # possible configuration values for later
        self._purchase_rate = purchase_rate_limit

    # buy and sell functionality based on allowed purchase rate, outside influence,
    def update(self, graph, available_stocks, id_to_broker_map, stocks, date):
        # include transaction, update status
        self.assess_portfolio_risk(stocks, date)
        neighbor_factor1 = self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)
        neighbor_factor2 = self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)

        transaction_count = 0
        while (self.current_risk < self.preferred_risk_minimum or self.current_risk > self.preferred_risk_maximum or random.random() < .4) and transaction_count < 20:
            # is there a better way to choose then randomly selecting higher or lower risk
            # if large gap, pull from back, if small from front half?
            transaction_count += 1
            if self.preferred_risk_minimum > self.current_risk and self.money > 100:
                # assess risk of every stock available and sort from lowest to highest risk
                risk_dict = dict()
                for stock in available_stocks:
                    risk_dict[stock] = self.assess_risk(stock, stocks, date) + neighbor_factor1[stock]
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)

                # select from stocks whose stock assessment put them in the risk min / 100 top percent of stocks
                sorted_list = list(filter(lambda a: stocks[a].date_to_price[date] < self.money, sorted_list))
                lower_limit = min(math.floor(len(sorted_list) * self.risk_percentile), len(sorted_list)-1)
                plan = random.sample(list(sorted_list[lower_limit:]), 1)[0]

                if len(sorted_list) < 2:
                    print(sorted_list)


                # buy stock and increment quantity in portfolio
                self.money -= stocks[plan].date_to_price[date]
                self.portfolio[plan] += 1

            elif self.preferred_risk_maximum < self.current_risk or self.money < 100:  # if above preferred risk or running out of money or noise, sell
                risk_dict = dict()
                for stock in self.portfolio:
                    if self.portfolio[stock] > 0:
                        risk_dict[stock] = self.assess_risk(stock, stocks, date) + neighbor_factor2[stock]

                if len(risk_dict) == 0:
                    break  # nothing can be done to lower the risk if they don't own any stocks
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)
                lower_limit = math.floor(len(sorted_list) * self.risk_percentile)
                plan = random.sample(list(sorted_list[lower_limit:]), 1)[0]

                self.money += stocks[plan].date_to_price[date]
                self.portfolio[plan] -= 1

            elif self.money > 100:  # buy more stocks
                # assess risk of every stock available and sort from lowest to highest risk
                risk_dict = dict()
                for stock in available_stocks:
                    risk_dict[stock] = self.assess_risk(stock, stocks, date) + neighbor_factor1[stock]
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)

                # select from stocks whose stock assessment put them in the risk min / 100 top percent of stocks
                sorted_list = list(filter(lambda a: stocks[a].date_to_price[date] < self.money, sorted_list))
                lower_limit = min(math.floor(len(sorted_list) * self.risk_percentile), len(sorted_list)-1)
                plan = random.sample(list(sorted_list[lower_limit:]), 1)[0]

                if len(sorted_list) < 2:
                    print(sorted_list)


                # buy stock and increment quantity in portfolio
                self.money -= stocks[plan].date_to_price[date]
                self.portfolio[plan] += 1


            self.assess_portfolio_risk(stocks, date)
        # print(self.id, self.current_risk)

        # update public status
        if self.adjust_influence:
            for neighbor in self.in_neighbors:
                self.in_neighbors[neighbor] += .0005 * np.sign(id_to_broker_map[neighbor].get_status(stocks, date) - self._neighbor_history[neighbor])
                self._neighbor_history[neighbor] = id_to_broker_map[neighbor].get_status(stocks, date)

    def get_status(self, stocks, date):
        """
        Calculates the value of the broker's stock portfolio given the status of the stocks on a given date assuming the
            broker were to sell immediately
        :param stocks: the stock pandas dataframe
        :param date: date in format 'MM/DD/YYYY'
        :return: float of current value of broker
        """

        portfolio_value = self.money
        for ticker in self.portfolio:
            portfolio_value += self.portfolio[ticker] * stocks[ticker].date_to_price[date]

        return portfolio_value

    def get_neighbor_factor(self, graph, available_stocks, id_to_broker_map, stocks, date):
        stock_dict = collections.defaultdict(lambda: 0)
        # iterate through graph getting neighbors and values - simple weighting?
        for n in self.in_neighbors:
            for stock in available_stocks:
                stock_dict[stock] += self.in_neighbors[n] * id_to_broker_map[n].assess_risk(stock, stocks, date)
        return stock_dict  # dictionary of neighbor risk assessment of each stock in the available stocks

    def assess_risk(self, ticker, stocks, date):
        """
        Assess the risk using a fat-tailed distribution formula.
        @param ticker: ticker for stock to assess
        @param stocks: dataframe of stock data
        @param date: date in format MM/DD/YYYY

        Returns:
            float: Risk assessment for a single stock.
        """

        # compute total equity for a given stock
        current_amount = self.portfolio[ticker] * stocks[ticker].date_to_price[date]
        portfolio_allocation = (current_amount + stocks[ticker].date_to_price[date]) / (
                    self.get_status(stocks, date) - self.money + stocks[ticker].date_to_price[date])
        portfolio_volume = self.portfolio[ticker]

        low = stocks[ticker].date_to_52low[date]
        high = stocks[ticker].date_to_52high[date]
        fifty_day_average = stocks[ticker].date_to_50day_avg[date]

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
        O(N)
        """
        portfolio_allocations, volumes, forward_eps_list, earnings_growth_list, high_list, low_list, dividend_ratio_list, fifty_day_average_list = list(), list(), list(), list(), list(), list(), list(), list()

        for ticker in self.portfolio:
            current_amount = self.portfolio[ticker] * stocks[ticker].date_to_price[date]
            portfolio_allocation = (current_amount) / (self.get_status(stocks, date) - self.money) if self.portfolio[ticker] != 0 else 0

            portfolio_allocations.append(portfolio_allocation)
            volumes.append(self.portfolio[ticker])
            forward_eps_list.append(stocks[ticker].forward_eps)
            earnings_growth_list.append(stocks[ticker].earnings_growth)
            high_list.append(stocks[ticker].date_to_52high[date])
            low_list.append(stocks[ticker].date_to_52low[date])
            dividend_ratio_list.append(stocks[ticker].dividend_ratio)
            fifty_day_average_list.append(stocks[ticker].date_to_50day_avg[date])

        k = 3  # Constant for fat-tailed distribution
        total_risk = 0
        for i in range(len(portfolio_allocations)):
            x = math.exp(abs(high_list[i] - low_list[i] - dividend_ratio_list[i] * fifty_day_average_list[i]) * -k)  # volatility
            a = portfolio_allocations[i] * forward_eps_list[i] * (
                    1 - earnings_growth_list[i])  # likelihood
            individual_risk = a * x
            total_risk += abs(individual_risk * volumes[i]) # scale up just for easier intuitive sense for parameters

        self.current_risk = total_risk
        return total_risk
