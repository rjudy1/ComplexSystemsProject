import ast
import collections
from datetime import datetime, timedelta
import math
import random
import numpy as np


class Broker:
    def __init__(self, id: int, initial_money: float, risk_minimum: float = 0, neighbor_weight: float = 0, shift_influence: bool = False):
        self.id = id
        self.money = initial_money
        self.portfolio = collections.defaultdict(lambda: 0)  # stock to quantity dictionary

        self.in_neighbors = dict()
        self.neighbor_weight = neighbor_weight

        self.preferred_risk_minimum = risk_minimum
        self.preferred_risk_maximum = risk_minimum*1.15

        self.risk_percentile = self.preferred_risk_minimum / 10000 #math.exp(.0065*self.preferred_risk_minimum) - 1
        # if self.risk_percentile >= 1.0:
        #     self.risk_percentile = .99
        # print(self.preferred_risk_minimum, self.preferred_risk_maximum, self.risk_percentile)

        self.current_risk = 0
        # self.current_status = 0

        self.adjust_influence = shift_influence
        self._neighbor_history = collections.defaultdict(lambda: initial_money)  # how do we want to use this?

    # buy and sell functionality based on allowed purchase rate, outside influence,
    def update(self, graph, available_stocks, id_to_broker_map, stocks, date):
        # include transaction, update status
        self.assess_portfolio_risk(stocks, date)
        neighbor_factor1 = self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)
        neighbor_factor2 = self.get_neighbor_factor(graph, available_stocks, id_to_broker_map, stocks, date)

        transaction_count = 0
        # while (self.money > 500 or self.preferred_risk_maximum < self.current_risk or random.random() < .2) and transaction_count < 100:
        while (self.current_risk < self.preferred_risk_minimum or self.current_risk > self.preferred_risk_maximum or random.random() < .45) and transaction_count < 50:
            # is there a better way to choose then randomly selecting higher or lower risk
            # if large gap, pull from back, if small from front half?
            transaction_count += 1

            # if our current risk is greater than the preferred risk maximum, sell risky stocks
            if self.preferred_risk_maximum < self.current_risk:
                percentage_consider = .1  # what percent of the given stock to consider selling

                risk_dict = dict()
                for stock in self.portfolio:
                    if self.portfolio[stock] > 0:
                        risk_dict[stock] = self.assess_risk(stock, max(int(self.portfolio[stock]*percentage_consider), 1), stocks, date) * (1-self.neighbor_weight) + neighbor_factor2[stock] * self.neighbor_weight

                if len(risk_dict) == 0:
                    break  # nothing can be done to lower the risk if they don't own any stocks
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)
                lower_limit = math.floor(len(sorted_list) * self.risk_percentile)

                if len(sorted_list) == 0:
                    continue

                plan = random.sample(list(sorted_list[lower_limit:]), 1)[0]

                self.money += stocks[plan].date_to_price[date] * max(int(self.portfolio[plan]*percentage_consider), 1)
                self.portfolio[plan] -= max(int(self.portfolio[plan]*percentage_consider), 1)


            # if we're below the risk level we want to buy more
            elif self.preferred_risk_minimum > self.current_risk:
                # sell some low risk ones if possible
                percentage_consider = .85  # percent of the chosen stock to dump

                risk_dict = dict()
                for stock in self.portfolio:
                    if self.portfolio[stock] > 0:
                        risk_dict[stock] = self.assess_risk(stock, max(int(self.portfolio[stock]*percentage_consider), 1), stocks, date) * (1-self.neighbor_weight) + neighbor_factor2[stock] * self.neighbor_weight

                if len(risk_dict) == 0:
                    break  # nothing can be done to lower the risk if they don't own any stocks
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)
                upper_limit = math.ceil(len(sorted_list) * self.risk_percentile)

                if len(sorted_list) != 0:
                    plan = random.sample(list(sorted_list)[:upper_limit], 1)[0]

                    self.money += stocks[plan].date_to_price[date] * max(int(self.portfolio[plan]*percentage_consider), 1)
                    self.portfolio[plan] -= max(int(self.portfolio[plan]*percentage_consider), 1)

                percentage_consider = .05  # percent of money to consider spending

                # assess risk of every stock available and sort from lowest to highest risk
                risk_dict = dict()
                quant_dict = dict()
                for stock in available_stocks:
                    # spend at most 100 or percent of money on as many stocks as that money can buy or 1 if not full
                    quantity = int(max(max(self.money * percentage_consider, 500) / stocks[stock].date_to_price[date], 1))
                    quant_dict[stock] = quantity
                    risk_dict[stock] = self.assess_risk(stock, quantity, stocks, date) * (1 - self.neighbor_weight) + \
                                       neighbor_factor1[stock] * self.neighbor_weight

                # select from stocks whose stock assessment put them in the risk min / 100 top percent of stocks
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)
                sorted_list = list(filter(lambda a: stocks[a].date_to_price[date] < self.money, sorted_list))

                # no stocks to buy
                if len(sorted_list) == 0:
                    continue

                lower_limit = min(math.floor(len(sorted_list) * self.risk_percentile), len(sorted_list) - 1)
                plan = random.sample(list(sorted_list[lower_limit:]), 1)[0]

                # buy stock and increment quantity in portfolio
                self.money -= stocks[plan].date_to_price[date] * quant_dict[plan]
                self.portfolio[plan] += quant_dict[plan]


            # if running out of money, sell some random stocks
            elif self.money < 500:
                # print("out of money 100")
                percentage_consider = .03  # percent of the stock of the given stock to consider dumping

                risk_dict = dict()
                for stock in self.portfolio:
                    if self.portfolio[stock] > 0:
                        risk_dict[stock] = self.assess_risk(stock, max(int(self.portfolio[stock]*percentage_consider), 1), stocks, date) * (1-self.neighbor_weight) + neighbor_factor2[stock] * self.neighbor_weight

                if len(risk_dict) == 0:
                    break  # nothing can be done to lower the risk if they don't own any stocks
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)

                if len(sorted_list) == 0:
                    continue

                plan = random.sample(list(sorted_list), 1)[0]

                self.money += stocks[plan].date_to_price[date] * max(int(self.portfolio[plan]*percentage_consider), 1)
                self.portfolio[plan] -= max(int(self.portfolio[plan]*percentage_consider), 1)


            # random buys
            else:  # self.money > 100:
                # print("rand om buys 123")
                percentage_consider = .0002  # percent of the money to spend

                risk_dict = dict()
                quant_dict = dict()
                for stock in available_stocks:
                    quantity = int(max(max(self.money*percentage_consider, 200) / stocks[stock].date_to_price[date], 1))
                    quant_dict[stock] = quantity
                    risk_dict[stock] = self.assess_risk(stock, quantity, stocks, date) * (1-self.neighbor_weight) + neighbor_factor1[stock] * self.neighbor_weight
                sorted_list = sorted(risk_dict.keys(), key=risk_dict.get)

                # select from stocks whose stock assessment put them in the risk min / 100 top percent of stocks
                sorted_list = list(filter(lambda a: stocks[a].date_to_price[date] < self.money, sorted_list))

                if len(sorted_list) == 0:
                    continue

                plan = random.sample(list(sorted_list), 1)[0]

                # buy stock and increment quantity in portfolio
                self.money -= stocks[plan].date_to_price[date] * quant_dict[plan]
                self.portfolio[plan] += quant_dict[plan]

            self.assess_portfolio_risk(stocks, date)

        # if self.id % 19 == 0:
        #     print(self.id, self.current_risk, self.money)
        # print(transaction_count)

        # update public status
        # @INFLUENCE
        if self.adjust_influence:
            for neighbor in self.in_neighbors:
                self.in_neighbors[neighbor] += .002 * np.sign(id_to_broker_map[neighbor].get_status(stocks, date) - self._neighbor_history[neighbor])
                if sum(self.in_neighbors[n] for n in self.in_neighbors) > 1.1:
                    self.neighbor_weight += .005
                elif sum(self.in_neighbors[n] for n in self.in_neighbors) < .9:
                    self.neighbor_weight -= .005
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

    # @INFLUENCE
    def get_neighbor_factor(self, graph, available_stocks, id_to_broker_map, stocks, date):
        stock_dict = collections.defaultdict(lambda: 0)
        # iterate through graph getting neighbors and values - simple weighting?
        for n in self.in_neighbors:
            for stock in available_stocks:
                stock_dict[stock] += self.in_neighbors[n] * id_to_broker_map[n].assess_risk(stock, 1, stocks, date)
        return stock_dict  # dictionary of neighbor risk assessment of each stock in the available stocks

    def assess_risk(self, ticker, quantity, stocks, date):
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
        portfolio_allocation = (current_amount + stocks[ticker].date_to_price[date] * quantity) / (
                    self.get_status(stocks, date) - self.money + stocks[ticker].date_to_price[date] * quantity)
        portfolio_volume = self.portfolio[ticker]

        low = stocks[ticker].date_to_52low[date]
        high = stocks[ticker].date_to_52high[date]
        fifty_day_average = stocks[ticker].date_to_50day_avg[date]

        k = 3  # Constant for fat-tailed distribution

        x = math.exp(abs(high - low - stocks[ticker].dividend_ratio * fifty_day_average) * -k)  # Equation to express volatility
        a = portfolio_allocation * stocks[ticker].forward_eps * (
                    1 - stocks[ticker].earnings_growth)  # Equation to express likelihood (a weighted sum of significant metrics)

        return a * x # if not math.isnan(a*x) else 0

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
            portfolio_allocation = current_amount / (self.get_status(stocks, date) - self.money) if self.portfolio[ticker] != 0 else 0

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
        return self.current_risk
