import math
from collections import defaultdict

class Broker:
    def __init__(self, id: int, initial_money: float, risk_minimum: float=0, risk_maximum: float=math.inf,
                 purchase_rate_limit: int=0, neighbor_influence: dict=dict()):
        self.id = id
        self.money = initial_money
        self._purchase_rate = purchase_rate_limit
        self._stocks = defaultdict(lambda: {})  # should we limit how many of each stock is avaiable?
        self._neighbors = neighbor_influence # not sure this should be inside the broker, maybe need to pass in when buying/selling a
                                             # and maintain the global graph

    # buy and sell functionality based on allowed purchase rate, outside influence,


    # update function
