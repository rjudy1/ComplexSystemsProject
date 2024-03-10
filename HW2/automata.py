import collections
import random
from utils import get_x_from_px, display_2D_automata


def random_move():
    pass


def social_network_recommendation():
    pass

def rachael_move():
    pass

def connor_move():
    pass

def josh_move():
    pass

policies = {0: random_move, 1: social_network_recommendation, 2: rachael_move, 3: connor_move, 4: josh_move}


def simulate_automata(L, p, k, epochs, relocation_policy):
    '''
    :param L: length of side of square environment
    :param p: percent of environment occupied
    :param k: threshold of 8 neighbors occupied by agents of its own types for happiness
    :return:
    '''
    def is_happy(layer, i, j):
        neighbors = {(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)}
        counter = collections.Counter(layer[indi%L][indj%L] for (indi, indj) in neighbors)
        return counter[layer[i][j]] >= k

    # create square LxL environment
    time_series = [[[get_x_from_px([0, 1, 2], [1-p, p/2, p/2]) for i in range(L)] for _ in range(L)] for i in range(5)]

    agents = [i for i in range(L*L)]
    for epoch in range(epochs):
        agents = agents.shuffle()
        for agent in agents:
            i, j = agent / L, agent % L
            # move agent if not happy
            relocation_policy()

    display_2D_automata(time_series)

simulate_automata(40, .8, 5, 10_000)

