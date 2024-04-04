# maybe use this class for the statistics chosen??
import dataclasses
import math

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
class Stock:
    def __init__(self, ticker, price):
        self.ticker = ticker
        self.price = price

        self.price_history = list()
        # TODO: add whatever other stats are relevant
