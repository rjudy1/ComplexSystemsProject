"""
Authors: Rachael Judy, Connor Klein, Josh Smith
The `simulate_automata` function is the core part of the functionality. To run a simulation, call the function

simulate_automata(L=50, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=0, policy_parameters=[], display_flag=True)

with the desired parameters. L is the length of the square grid, alpha determines what percentage of the grid is
populated, k sets how many neighbors are needed to be completely happy, epochs sets how many epochs to run in a trial,
trials sets the number of trials, relocation_policy selects the policy to use (currently available are 0-4),
policy parameters is a list of the parameters for a policy in order, and display_flag determines whether to show
the visual view of the automata.

The available policies are currently as follows:
random_move (0): searches q empty spots at random and goes to one with greatest happiness
social_network_recommendation (1): based on a randomly selected network of n friends, take the best options from a pxp
    sized neighborhood around the friend and move there
rachael_move (2):
connor_move (3):
josh_move (4):

To add more functions, simply add to the policies dictionary and extend the code calling the policies. Each new function
should return a tuple representing the new position (i, j)

"""

import collections
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import get_x_from_px, display_2D_automata


def random_move(layer, agent: int, agent_positions: dict, k: int, q: int) -> tuple:
    """
    Base case of searching up to q empty positions and moves to first where happy or if not possible, happiest possible
    :param layer: current state of the cellular automata
    :param agent: agent whose position is being explored around
    :param agent_positions: dictionary of all agents to their current position
    :param q: number of empty positions to search
    :param k: required number of like neighbors to be happy
    :return: tuple (i, j) of recommended spot to move
    """
    # collect empty positions
    empty_positions = {agent_positions[agent]}
    for i in range(len(layer)):
        for j in range(len(layer[0])):
            if layer[i][j] == 0:
                empty_positions.add((i, j))

    # look over q empty spots for one which would provide maximum happiness
    best_spot, happiest = agent_positions[agent], -1
    for _ in range(q):
        potential_location = random.sample(list(empty_positions), 1)[0]
        happiness = get_happiness(layer, potential_location, layer[agent_positions[agent][0]][agent_positions[agent][1]], k)
        if happiness == 1:
            return potential_location
        if happiness > happiest:
            happiest = happiness
            best_spot = potential_location
        empty_positions.remove(potential_location)
    return best_spot


def social_network_recommendation(layer, agent: int, agent_positions: dict, k: int, n: int, p: int, q: int, friend_map) -> tuple:
    # assuming suitable means perfectly happy
    # we're going to yield instead of return so we can set up the neighbors
    # assign every agent a unique identifier from the start and a group of n friends
    # Use None as a sentinel value to trigger initialization
    curri, currj = agent_positions[agent]
    recommendations = set()
    for friend in friend_map[agent]:
        fi, fj = agent_positions[friend]
        for i in range(-p // 2, p // 2 + 1):
            for j in range(-p // 2, p // 2 + 1):
                if (layer[(fi + i) % len(layer)][(fj + j) % len(layer)] == 0
                        and get_happiness(layer, (fi + i, fj + j), layer[curri][currj], k) == 1):
                    recommendations.add(((fi + i) % len(layer), (fj + j) % len(layer)))
    if recommendations:
        # print(random.sample(list(recommendations), 1)[0])
        return random.sample(list(recommendations), 1)[0]
    else:
        return random_move(layer, agent, agent_positions, k, q)


def rachael_move():
    pass

def connor_move():
    pass

def josh_move():
    pass


def get_happiness(layer: list[list], position: tuple, desired_value: int, k: int):
    """
    Returns number from 0 to 1 representing the happiness of a value placed at a given position
    :param layer: current state of the automata
    :param position: position under consideration
    :param desired_value: type of space the agent wants to be near
    :param k: number of neighbors required for total happiness - neighbors wraparound
    :return: happiness = 1 if greater than or equal to k matching neighbors,
                            otherwise matching neighbors / 8 + empty neighbors / 32  - TODO: this can be changed
    """
    i, j = position
    neighbors = {(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1),
                 (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)}
    counter = collections.Counter(layer[indi % len(layer)][indj % len(layer)] for (indi, indj) in neighbors)
    if counter[desired_value] >= k:
        return 1
    else:
        return counter[desired_value] / 8 - counter[0] / 32


def rate_performance(layer: list[list], k):
    """
    Score the state of the system by summing every agents' happiness
    :param layer: current state of the overall automata
    :param k: number neighbors needed for complete happiness
    :return: float representing total happiness of system
    """
    return sum(sum(get_happiness(layer, (i, j), layer[i][j], k) if layer[i][j] != 0 else 0
                   for j in range(len(layer[0]))) for i in range(len(layer)))


def simulate_automata(L: int, alpha: float, k: int, epochs: int = 20, trials: int = 20,
                      relocation_policy: int = 0, policy_parameters: list = [], display_flag: bool = False):
    """
    :param L: length of side of square environment
    :param alpha: percent of environment occupied
    :param k: threshold of 8 neighbors occupied by agents of its own types for happiness
    :param epochs: number of epochs per trial
    :param trials: number of trials to run before reporting average and std deviation
    :param relocation_policy: integer selection of policy; 0 corresponds to random
    :param policy_parameters: list of parameter policies in order as specified by policy function
    :param display_flag: boolean indicating whether to display the visual of the time series for each trial
    :return: two collections, one of the average happiness at each epoch, one of the standard deviations at each epoch
    """

    global policies
    epoch_to_scores = collections.defaultdict(lambda: list())  # will contain the epoch to scores at this epoch
    for trial in range(trials):
        # create square LxL environment with desired percentage of empty, red, and blue squares
        time_series = [[[get_x_from_px([0, 1, 2], [1-alpha, alpha/2, alpha/2])
                         for _ in range(L)] for _ in range(L)]]
        epoch_to_scores[0].append(rate_performance(time_series[0], k))  # note initial score

        # create list to sample from for randomly moving an agent - agents are assigned index based on initial position
        agents = list(filter(lambda val: time_series[0][val // L][val % L] != 0, [i for i in range(L*L)]))
        agent_positions = {agent: (agent // L, agent % L) for agent in agents}

        # TODO: Add necessary initialization for each method
        if relocation_policy == 1:
            friend_map = {a: random.sample(agents, policy_parameters[0]) for a in agents}

        for epoch in range(epochs):
            random.shuffle(agents)
            for agent in agents:
                i, j = agent_positions[agent]
                # square does not contain an agent, ignore
                if time_series[-1][i][j] == 0:
                    continue

                # TODO: UPDATE THESE ACCORDING TO USAGE: if stabilized format for functions, can just make a single call
                # move agent if not happy
                if get_happiness(time_series[-1], (i,j), time_series[-1][i][j], k) != 1:
                    # move function returns the new location
                    if relocation_policy == 0:
                        new_i, new_j = policies[0](layer=time_series[-1], agent=agent, agent_positions=agent_positions,
                                                   k=k, q=policy_parameters[0])
                    elif relocation_policy == 1:
                        new_i, new_j = policies[1](layer=time_series[-1],  agent=agent, agent_positions=agent_positions,
                                                   k=k, n=policy_parameters[0], p=policy_parameters[1], q=policy_parameters[2],
                                                   friend_map=friend_map)
                    elif relocation_policy == 2:
                        policies[2]()
                    elif relocation_policy == 3:
                        policies[3]()
                    elif relocation_policy == 4:
                        policies[4]()

                    # move to new position and adjust visual as well
                    agent_positions[agent] = (new_i, new_j)
                    environment_copy = deepcopy(time_series[-1])
                    environment_copy[new_i][new_j], environment_copy[i][j] = environment_copy[i][j], 0
                    time_series.append(environment_copy)

            # at end of each epoch, score the performance
            epoch_to_scores[epoch + 1].append(rate_performance(time_series[-1], k))

        if display_flag:
            display_2D_automata(time_series)
    return epoch_to_scores


policies = {0: random_move, 1: social_network_recommendation, 2: rachael_move, 3: connor_move, 4: josh_move}

# test random - control case
results_base = simulate_automata(L=40, alpha=.9, k=5, epochs=20, trials=20, relocation_policy=0, policy_parameters=[100], display_flag=False)

# test social
results_n5p3 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 5, 100], display_flag=False)
results_n5p5 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 5, 100], display_flag=False)
results_n10p3 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 5, 100], display_flag=False)
results_n10p5 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 5, 100], display_flag=False)
results_n20p3 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 5, 100], display_flag=False)
results_n20p5 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 5, 100], display_flag=False)



# TODO: add plotting code for averages/standard deviations
averages = [
    [np.average(results_base[row]) for row in results_base],
    [np.average(results_n5p3[row]) for row in results_n5p3],
    [np.average(results_n5p5[row]) for row in results_n5p5],
    [np.average(results_n10p3[row]) for row in results_n10p3],
    [np.average(results_n10p5[row]) for row in results_n10p5],
    [np.average(results_n20p3[row]) for row in results_n20p3],
    [np.average(results_n20p5[row]) for row in results_n20p5],
]

std_deviations = [
    [np.std(results_base[row]) for row in results_base],
    [np.std(results_n5p3[row]) for row in results_n5p3],
    [np.std(results_n5p5[row]) for row in results_n5p5],
    [np.std(results_n10p3[row]) for row in results_n10p3],
    [np.std(results_n10p5[row]) for row in results_n10p5],
    [np.std(results_n20p3[row]) for row in results_n20p3],
    [np.std(results_n20p5[row]) for row in results_n20p5],
]

labels = ['Random Move with q=100', 'Friend Recommendation with n=5, p=3','Friend Recommendation with n=5, p=5',
          'Friend Recommendation with n=10, p=3','Friend Recommendation with n=10, p=5',
          'Friend Recommendation with n=20, p=3','Friend Recommendation with n=20, p=5',]

# Plotting
plt.figure(figsize=(10, 6))

for i in range(len(averages)):
    plt.errorbar(
        np.arange(1, 22),
        averages[i],
        yerr=std_deviations[i],
        label=f'{labels[i]}',
        alpha=0.7
    )

# Customize the plot
plt.title('Time Series with Averages and Standard Deviation Bars')
plt.xlabel('Time Points')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(f'images/time_series1.png')