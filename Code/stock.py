# maybe use this class for the statistics chosen??
import dataclasses


class Stock:
    def __init__(self, ticker, price):
        self.ticker = ticker
        self.price = price

        self.price_history = list()
        # TODO: add whatever other stats are relevant
