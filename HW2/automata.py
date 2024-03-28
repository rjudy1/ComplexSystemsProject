"""
Authors: Rachael Judy, Connor Klein, Josh Smith
The `simulate_automata` function is the core part of the functionality. To run a simulation, call the function

simulate_automata(L=50, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=0, policy_parameters=[], display_flag=True)
    L: length of side of square environment
    alpha: percent of environment occupied
    k: threshold of 8 neighbors occupied by agents of its own types for happiness
    epochs: number of epochs per trial
    trials: number of trials to run before reporting average and std deviation
    relocation_policy: integer selection of policy; 0 corresponds to random
    policy_parameters: list of parameter policies in order as specified by policy function
    display_flag: boolean indicating whether to display the visual of the time series for each trial
    return: two collections, one of the average happiness at each epoch, one of the standard deviations at each epoch


The available policies are currently as follows:
random_move (0): searches q empty spots at random and goes to one with greatest happiness
social_network_recommendation (1): based on a randomly selected network of n friends, take the best options from a pxp
    sized neighborhood around the friend and move there
rachael_move (2): Recommends a position based on connected matching agents within p squares + small percent beta
    of differing neighbors, simulating asking neighbors of the same type for recommendations around them
connor_move (3): Recommends a position based on hotspots in the area. Agents have increased probability to go to a site
    near multiple hotspots
josh_move (4):

To add more functions, simply add to the policies dictionary and extend the code calling the policies. Each new function
should return a tuple representing the new position (i, j)

idea: trade positions if both will be happier or if one is happier and the other is equally happy (will require change of position swap to actual value instead of zero)

"""

import collections
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import get_x_from_px, display_2D_automata, save_2D_automata


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
    for _ in range(min(q, len(empty_positions))):
        potential_location = random.sample(list(empty_positions), 1)[0]
        happiness = get_happiness(layer, potential_location, layer[agent_positions[agent][0]][agent_positions[agent][1]], k)
        if happiness == 1:
            return potential_location
        if happiness > happiest:
            happiest = happiness
            best_spot = potential_location
        empty_positions.remove(potential_location)
    return best_spot


def social_network_recommendation(layer, agent: int, agent_positions: dict, k: int, n: int, p: int, q: int,
                                  friend_map: dict) -> tuple:
    """
    Recommends a position to move based on friends of agent reporting spots agent would be happy
    :param layer: current state of the system
    :param agent: id of agent to move
    :param agent_positions: id to position map
    :param k: number of neighbors required for complete happiness
    :param n: number of friends
    :param p: side length of grid around friend to explore
    :param q: parameter for random move as backup
    :param friend_map: preinitialized map of agent id to n friends
    :return: tuple (i, j) of recommended spot to move
    """
    curri, currj = agent_positions[agent]
    recommendations = set()
    current_happiness = get_happiness(layer, (curri, currj), layer[curri][currj], k)
    # go through friends pxp neighborhoods
    for friend in friend_map[agent]:
        fi, fj = agent_positions[friend]
        for i in range(-p // 2, p // 2 + 1):
            for j in range(-p // 2, p // 2 + 1):
                if (layer[(fi + i) % len(layer)][(fj + j) % len(layer)] == 0
                        and get_happiness(layer, (fi + i, fj + j), layer[curri][currj], k) > current_happiness):
                    recommendations.add(((fi + i) % len(layer), (fj + j) % len(layer)))

    # look over q empty spots for one which would provide maximum happiness
    best_spot, happiest = agent_positions[agent], -1
    for _ in range(min(q, len(recommendations))):
        potential_location = random.sample(list(recommendations), 1)[0]
        happiness = get_happiness(layer, potential_location, layer[curri][currj], k)
        if happiness == 1:
            return potential_location
        if happiness > happiest:
            happiest = happiness
            best_spot = potential_location
        recommendations.remove(potential_location)
    if best_spot != agent_positions[agent]:
        return best_spot

    # # if friends had ideas, sample from this; otherwise, move randomly
    # if recommendations:
    #     return random.sample(list(recommendations), 1)[0]
    return random_move(layer, agent, agent_positions, k, q)


def rachael_move(layer, agent: int, agent_positions: dict, k: int, w: int, beta: float, q: int):
    """
    Recommends a position based on connected matching agents within p squares + small percent beta of differing neighbors
    :param layer: current state of the system
    :param agent: id of agent to move
    :param agent_positions: id to position map
    :param k: number of neighbors required for complete happiness
    :param p: number of moves away from current position to look in own matches
    :param beta: float representing probability of asking a non-matching adjacent for recommendations
    :param q: parameter for random move as backup
    :param friend_map:
    :return: tuple (i, j) of recommended spot to move
    """
    # explore with BFS
    L = len(layer)
    visited, to_visit, available_positions = set(), [(*agent_positions[agent], 0)], set()
    while to_visit and to_visit[0][2] < w:
        i, j, depth = to_visit.pop(0)
        for di, dj in {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}:
            if ((layer[(i+di) % L][(j+dj) % L] == layer[i][j] or
                layer[(i+di) % L][(j+dj) % L] != layer[i][j] and layer[(i+di) % L][(j+dj) % L] != 0 \
                and get_x_from_px([0, 1], (1-beta, beta)))
                    and ((i+di)%L, (j+dj) % L) not in visited):
                to_visit.append(((i+di) % L, (j+dj) % L, depth+1))
                visited.add(((i+di)%L, (j+dj)%L))
            elif layer[(i+di) % L][(j+dj) % L] == 0 and get_happiness(layer, ((i+di) % L, (j+dj) % L), layer[i][j], k) == 1:
                available_positions.add(((i+di) % L, (j+dj) % L))

    if available_positions:
        return random.sample(list(available_positions), 1)[0]
    return random_move(layer, agent, agent_positions, k, q)


def connor_move(layer, agent: int, agent_positions: dict, k: int, q: int, R: int, hotspots: list):
    """
    Recommends a position based on hotspots in the area. Agents have increased probability to go to a site near multiple hotspots
    :param layer: current state of the system
    :param agent: id of agent to move
    :param agent_positions: id to position map
    :param k: number of neighbors required for complete happiness
    :param q: parameter for random move as backup
    :param R: parameter of radius outward to check, where the center is 1, 2 is one tile away, etc
    :param centers: positions of centers
    :return: tuple (i, j) of recommended spot to move
    """
    L = len(layer)
    available_positions = []
    probablity_distrubution = []
    
    for site in hotspots:
        for r in range(1, R):
            xy = []
            for i in range(site[0] - r, site[0] + r + 1):
                xy.append(((i + L) % L, (site[1] - r + L) % L))
                xy.append(((i + L) % L, (site[1] + r + L) % L))

            for j in range(site[1] - r + 1, site[1] + r):
                xy.append(((site[0] - r + L) % L, (j + L) % L))
                xy.append(((site[0] + r + L) % L, (j + L) % L))
            
            for x,y in xy:
                if layer[x][y] == 0 and get_happiness(layer, (x,y), layer[site[0]][site[1]], k) == 1:
                    available_positions.append((x,y))
                    probablity_distrubution.append(1/r)
            
    if not available_positions:
        return random_move(layer, agent, agent_positions, k, q)
    else:
        # Choose a new place based on the radius as a probility
        probablity_distrubution = np.array(probablity_distrubution)
        probablity_distrubution = probablity_distrubution / np.sum(probablity_distrubution)
        index = np.random.choice(len(available_positions), 1, replace=False, p=probablity_distrubution)[0]
        return available_positions[index]
    
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
                            otherwise matching neighbors / 8 + empty neighbors / 64  - TODO: this can be changed
    """
    i, j = position
    neighbors = {(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1),
                 (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)}
    counter = collections.Counter(layer[indi % len(layer)][indj % len(layer)] for (indi, indj) in neighbors)
    if counter[desired_value] >= k:
        return 1
    else:
        return counter[desired_value] / 8 + counter[0] / 64


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
                      relocation_policy: int = 0, policy_parameters: list = [], display_flag: bool = False, save_flag: bool = False):
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
        if relocation_policy == 3:
            hotspots = [np.random.choice(L, policy_parameters[2], replace=False), np.random.choice(L, policy_parameters[2], replace=False)]
            hotspots = list(zip(hotspots[0], hotspots[1]))
            
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
                        new_i, new_j = policies[2](layer=time_series[-1], agent=agent, agent_positions=agent_positions,
                                                   k=k, w=policy_parameters[0], beta=policy_parameters[1], q=policy_parameters[2])
                    elif relocation_policy == 3:
                        new_i, new_j = policies[3](layer=time_series[-1], agent=agent, agent_positions=agent_positions,
                                                   k=k, q=policy_parameters[0], R=policy_parameters[1], hotspots = hotspots)
                    elif relocation_policy == 4:
                        policies[4]()

                    # move to new position and adjust visual as well
                    environment_copy = deepcopy(time_series[-1])
                    if (new_i, new_j) != agent_positions[agent]:
                        environment_copy[new_i][new_j], environment_copy[i][j] = environment_copy[i][j], 0
                    time_series.append(environment_copy)
                    agent_positions[agent] = (new_i, new_j)

            # at end of each epoch, score the performance
            epoch_to_scores[epoch + 1].append(rate_performance(time_series[-1], k))

        if display_flag and trial == 0:
            display_2D_automata(time_series, f'policy{relocation_policy}_')
        elif save_flag and trial == 0:
            save_2D_automata(time_series, f'policy{relocation_policy}')
    return epoch_to_scores


def plot_averages_with_errorbars(averages, std_deviations, labels, filename,
                                 title='Time Series with Averages and Standard Deviation Bars'):
    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(len(averages)):
        plt.errorbar(
            np.arange(0, len(averages[i])),
            averages[i],
            yerr=std_deviations[i],
            label=f'{labels[i]}',
            alpha=0.7
        )

    # Customize the plot
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{filename}')
    plt.show()


policies = {0: random_move, 1: social_network_recommendation, 2: rachael_move, 3: connor_move, 4: josh_move}

# test random - control case
results_base = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=0, policy_parameters=[100], display_flag=False)

# # test social
# averages1 = [[np.average(results_base[row]) for row in results_base]]
# std_deviations1 = [[np.std(results_base[row]) for row in results_base]]
# labels1 = ['Random Move with q=100']
# for n in [5, 10, 20]:
#     for p in [3, 5]:
#         result = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=30, relocation_policy=1, policy_parameters=[n, p, 100], display_flag=False)
#         averages1.append([np.average(result[row]) for row in result])
#         std_deviations1.append([np.std(result[row]) for row in result])
#         labels1.append(f'Friend Recommendation with n={n}, p={p}')
# plot_averages_with_errorbars(averages1, std_deviations1, labels1, "policies01.png", "Random move policy compared to friend recommendation policy")


# test rachael policy
# averages2 = [[np.average(results_base[row]) for row in results_base]]
# std_deviations2 = [[np.std(results_base[row]) for row in results_base]]
# labels2 = ['Random move with q=100']
# for beta in [.1, .2]:
#     for w in [5, 10, 20]:
#         result = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=2, policy_parameters=[w, beta, 100], display_flag=False)
#         averages2.append([np.average(result[row]) for row in result])
#         std_deviations2.append([np.std(result[row]) for row in result])
#         labels2.append(f'Search neighborhood with w={w}, beta={beta}')
# plot_averages_with_errorbars(averages2, std_deviations2, labels2, "policies02.png", "Random move policy compared to search neighborhood policy")

# test connor's policy
averages3 = [[np.average(results_base[row]) for row in results_base]]
std_deviations3 = [[np.std(results_base[row]) for row in results_base]]
labels2 = ['Random move with q=100']
for sites in [2,3]:
    for radius in [5,10,15]:
        result = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=3, policy_parameters=[100, radius, sites], display_flag=False, save_flag=True)
        averages3.append([np.average(result[row]) for row in result])
        std_deviations3.append([np.std(result[row]) for row in result])
        labels2.append(f'Search around {sites} site(s) with radius = {radius}')
plot_averages_with_errorbars(averages3, std_deviations3, labels2, "policies03.png", "Random move policy compared to Site Search Policy")

