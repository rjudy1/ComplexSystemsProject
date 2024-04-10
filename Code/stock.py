# maybe use this class for the statistics chosen??
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import numpy as np

@dataclass
class Stock:
    def __init__(self, ticker, forward_eps, earnings_growth, dividend_ratio, dates, prices):
        self.ticker = ticker

        self.forward_eps = forward_eps
        self.earnings_growth = earnings_growth
        self.dividend_ratio = dividend_ratio

        self.dates = dates
        self.prices = prices

        self.date_to_price = {d: p for d, p in zip(dates, prices)}

        self.date_to_52high = dict()
        self.date_to_52low = dict()
        self.date_to_50day_avg = dict()
        start = datetime.strptime(dates[0], "%m/%d/%Y")
        current_date = datetime.strptime(dates[0], "%m/%d/%Y")
        end_date = datetime.strptime("12/31/2023", "%m/%d/%Y")        # Converting a to string in the desired format (YYYYMMDD) using strftime

        while current_date < end_date:
            if current_date.strftime("%m/%d/%Y") not in self.date_to_price:
                previous_date = current_date - timedelta(days=1)
                self.date_to_price[current_date.strftime("%m/%d/%Y")] = self.date_to_price[previous_date.strftime("%m/%d/%Y")]
            current_date += timedelta(days=1)

        # create list of every price and roll over it, assigning based on date window
        curr = start
        prices_list = list()
        while curr < end_date:
            prices_list.append(self.date_to_price[curr.strftime('%m/%d/%Y')])
            curr += timedelta(days=1)
        curr = start
        idx = 0
        while curr < end_date:
            front = max(idx-49, 0)
            self.date_to_50day_avg[curr.strftime('%m/%d/%Y')] = np.average(prices_list[front:idx+1])

            front52 = max(idx - 52*7 + 1, 0)
            self.date_to_52high[curr.strftime('%m/%d/%Y')] = max(prices_list[front52:idx+1])
            self.date_to_52low[curr.strftime('%m/%d/%Y')] = min(prices_list[front52:idx+1])
            idx += 1
            curr += timedelta(days=1)


'''
# Original risk code -> working code has been transferred into broker 

def assess_risk(portfolio_allocation, forward_eps, earnings_growth, high, low, dividend_ratio, fifty_day_average):
    """
    Assess the risk using a fat-tailed distribution formula.

    Args:
        portfolio_allocation (float): Portfolio allocation for stocks.
        forward_eps (float): Forward EPS values for stocks.
        earnings_growth (float): Earnings quarterly growth for stocks.
        high (float): 52-week high.
        low (float): 52-week low.
        dividend_ratio (float): Dividend ratio (expected return).
        fifty_day_average (float): Average value of the stock over the last 50 days.

    Returns:
        float: Risk assessment for a single stock.

    """
    k = 3  # Constant for fat-tailed distribution
    x = math.exp(abs(high - low - dividend_ratio * fifty_day_average))  # Equation to express volatility
    a = portfolio_allocation * forward_eps * (1 - earnings_growth)  # Equation to express likelihood (a weighted sum of significant metrics)

    return a * (x ** -k)

# Example usage:
portfolio_allocation = 0.2  # Portfolio allocation for stock
forward_eps = 1.5  # Forward EPS value for stock
earnings_growth = 0.1 # Earnings quarterly growth for stocks
high = 100  # 52-week high
low = 50  # 52-week low
fifty_day_average = 75  # Average value of the stock over the last 50 days
dividend_ratio = 0.1  # Dividend ratio (expected return)

# FUTURE WORK
# Change dividend ratio to trailingAnnualDividendRate?
# PEG ratio over earningsQuarterlyGrowth?

risk = assess_risk(portfolio_allocation, forward_eps, earnings_growth, high, low, dividend_ratio, fifty_day_average)
print("Risk assessment:", risk)

def assess_portfolio_risk(portfolio_allocation, forward_eps_list, earnings_growth_list, high_list, low_list, dividend_ratio_list, fifty_day_average_list):
    """
    Assess the risk of the entire portfolio using a fat-tailed distribution formula.

    Args:
        portfolio_allocation (list of float): Portfolio allocation for each stock.
        forward_eps_list (list of float): Forward EPS values for each stock.
        earnings_growth_list (list of float): Earnings quarterly growth for each stock.
        high_list (list of float): 52-week high for each stock.
        low_list (list of float): 52-week low for each stock.
        dividend_ratio_list (list of float): Dividend ratio (expected return) for each stock.
        fifty_day_average_list (list of float): Average value of each stock over the last 50 days.

    Returns:
        float: Risk assessment for the entire portfolio.

    """
    k = 3  # Constant for fat-tailed distribution
    total_risk = 0
    for i in range(len(portfolio_allocation)):
        x = math.exp(abs(high_list[i] - low_list[i] - dividend_ratio_list[i] * fifty_day_average_list[i]))  # Equation to express volatility
        a = portfolio_allocation[i] * forward_eps_list[i] * (1 - earnings_growth_list[i])  # Equation to express likelihood
        individual_risk = a * (x ** -k)
        total_risk += individual_risk

    return total_risk

# Example usage:
portfolio_allocation = [0.2, 0.3, 0.5]  # Portfolio allocation for each stock
forward_eps_list = [1.5, 2.0, 1.8]  # Forward EPS values for each stock
earnings_growth_list = [0.1, 0.05, 0.08]  # Earnings quarterly growth for each stock
high_list = [100, 120, 90]  # 52-week high for each stock
low_list = [50, 60, 40]  # 52-week low for each stock
fifty_day_average_list = [75, 80, 70]  # Average value of each stock over the last 50 days
dividend_ratio_list = [0.1, 0.05, 0.08]  # Dividend ratio (expected return) for each stock

portfolio_risk = assess_portfolio_risk(portfolio_allocation, forward_eps_list, earnings_growth_list, high_list, low_list, dividend_ratio_list, fifty_day_average_list)
print("Portfolio risk assessment:", portfolio_risk)

#Alternatively, you can just take the sum of all the individual risks of stocks
'''