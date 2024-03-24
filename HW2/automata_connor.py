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

idea: trade positions if both will be happier (will require change of position swap to actual value instead of zero)

"""

import collections
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random
from HW2.utils_connor import get_x_from_px, save_start_and_end


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
    for friend in friend_map[agent]:
        fi, fj = agent_positions[friend]
        for i in range(-p // 2, p // 2 + 1):
            for j in range(-p // 2, p // 2 + 1):
                if (layer[(fi + i) % len(layer)][(fj + j) % len(layer)] == 0
                        and get_happiness(layer, (fi + i, fj + j), layer[curri][currj], k) == 1):
                    recommendations.add(((fi + i) % len(layer), (fj + j) % len(layer)))
    if recommendations:
        return random.sample(list(recommendations), 1)[0]
    else:
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


def connor_move(time_series, agent: int, agent_positions: dict, desired_value: int, heat_map: dict, r_relocate_g: float):
    """
    there exist places in the map that are more desirable than other locations,
    except one group has the ability to kick out the other group
    
    :param layer: current state of the system
    :param agent: id of agent to move
    :param agent_positions: id to position map
    :param hot_spots: dict of position with desirabilty %
    """
    layer = time_series[-1]
    L = len(layer)
    if desired_value == 1:
        # updating the heatmap to add probabilites
        updated_heat_map = heat_map.copy()
        
        # first where the greens are
        mask = np.ma.masked_equal(layer, 2).mask
        updated_heat_map[mask] *= r_relocate_g
        
        # second where the 1's are, can't go there
        mask = np.ma.masked_equal(layer, 1).mask
        updated_heat_map[mask] = 0.0
    
    if desired_value == 2:
        # updating the heatmap to add probabilites
        updated_heat_map = heat_map.copy()
        
        # map of where the 0's are, can't move anywhere else
        mask = np.ma.masked_where((layer == 1) | (layer == 2), layer).mask
        updated_heat_map[mask] = 0.0
        
    new_i, new_j = choose_from_heatmap(updated_heat_map)
    if desired_value == 1 and layer[new_i][new_j] == 2:
        for agent_to_evict, position in agent_positions.items():
            if position == (new_i, new_j):
                # kick out the existing agent to somewhere random
                green_eviction = random_move(layer, agent_to_evict, agent_positions, k=0, q=10)
                # secondary update to the env, but it happens first
                update_enviroment(time_series, green_eviction[0], green_eviction[1], new_i, new_j, agent_positions, agent_to_evict)
                break

    return new_i, new_j

def josh_move():
    pass

def create_heatmap(L: int, hotspot_amount: int, radius: float):
    hotspot_positions = [np.random.choice(L, hotspot_amount, replace=False), np.random.choice(L, hotspot_amount, replace=False)]
    hotspot_positions = list(zip(hotspot_positions[0], hotspot_positions[1]))
    
    heat_map = np.zeros((L,L))
    # create heatmap
    for position in hotspot_positions:
        for index, cell in np.ndenumerate(heat_map):
            row, col = index
            dx = abs(position[0] - row)
            dy = abs(position[1] - col)
            dx = min(dx, L-dx)
            dy = min(dy, L-dy)
            
            heat_map[row][col] = cell + np.exp(-np.sqrt(dx**2 + dy**2) / radius) #e^-x/radius
            
    # set hotspots to zero
    for position in hotspot_positions:
        heat_map[position[0],position[1]] = 0.0
    # normalize
    heat_map = heat_map / np.sum(heat_map)
    
    # for funzies
    # plot_heat_map(heat_map)
    # if hotspot_amount == 1:
    #     hotspot_positions = [hotspot_positions]
    return heat_map, hotspot_positions

def plot_heat_map(heat_map):
    plt.imshow(heat_map, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add colorbar to show scale
    plt.title('Heatmap of 2D NumPy Array')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

def choose_from_heatmap(heatmap):
    flattened_heatmap = heatmap.flatten()
    flattened_heatmap = flattened_heatmap / np.sum(flattened_heatmap)
    index = np.random.choice(len(flattened_heatmap), 1, p=flattened_heatmap)
    row, col = np.unravel_index(index, heatmap.shape) #returns a tuple
    return row[0], col[0]

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

def get_happyness_connor_move(layer_length, hotspot_positions, agent_position, closness):
    distance = np.sqrt(layer_length**2 + layer_length**2)
    for position in hotspot_positions:
        dx = abs(position[0] - agent_position[0])
        dy = abs(position[1] - agent_position[1])
        dx = min(dx, layer_length-dx)
        dy = min(dy, layer_length-dy)
        
        # only move if your not close enough
        distance = min(np.sqrt(dx**2 + dy**2), distance)
        
    if closness >  distance:
        return 1
    else:
        return 1/distance
    
def rate_performance(layer: list[list], k):
    """
    Score the state of the system by summing every agents' happiness
    :param layer: current state of the overall automata
    :param k: number neighbors needed for complete happiness
    :return: float representing total happiness of system
    """
    return sum(sum(get_happiness(layer, (i, j), layer[i][j], k) if layer[i][j] != 0 else 0
                   for j in range(len(layer[0]))) for i in range(len(layer)))
    
def rate_performance_policy_3(layer, hotspots, closness):
    ones = [(i,j) for i in range(len(layer)) for j in range(len(layer)) if layer[i][j] == 1]
    twos = [(i,j) for i in range(len(layer)) for j in range(len(layer)) if layer[i][j] == 2]
    
    one_sum = sum(get_happyness_connor_move(len(layer), hotspots, position, closness) for position in ones)
    two_sum = sum(get_happyness_connor_move(len(layer), hotspots, position, closness) for position in twos)
    
    return [one_sum, two_sum]

def update_enviroment(time_series, new_i, new_j, i, j, agent_positions, agent):
    """
    Score the state of the system by summing every agents' happiness
    :param enviroment: current state of the overall automata, the layer
    :param new_i: the new x of the agent
    :param new_j: the new y of the agent
    :param i: the old x of the agent
    :param j: the old y of the agent
    :param agent: id of agent to move
    :param agent_positions: id to position map
    
    :return: the new eniviroment
    """
    environment_copy = deepcopy(time_series[-1])
    if (new_i, new_j) != agent_positions[agent]:
        environment_copy[new_i][new_j], environment_copy[i][j] = environment_copy[i][j], environment_copy[new_i][new_j]
    time_series.append(environment_copy)
    agent_positions[agent] = (new_i, new_j)

def simulate_automata(L: int, alpha: float, k: int, epochs: int = 20, trials: int = 20,
                      relocation_policy: int = 0, policy_parameters: list = [], display_flag: bool = False,
                      happiness_function = get_happiness):
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
        

        # create list to sample from for randomly moving an agent - agents are assigned index based on initial position
        agents = list(filter(lambda val: time_series[0][val // L][val % L] != 0, [i for i in range(L*L)]))
        agent_positions = {agent: (agent // L, agent % L) for agent in agents}

        # TODO: Add necessary initialization for each method
        if relocation_policy == 1:
            friend_map = {a: random.sample(agents, policy_parameters[0]) for a in agents}
        if relocation_policy == 3:
            heat_map, hotspot_locations = create_heatmap(L, policy_parameters[0], policy_parameters[1])
            # plot_heat_map(heat_map)
            for location in hotspot_locations:
                for agent, position in agent_positions.items():
                    if position == location:
                        eviction = random_move(time_series[-1], agent, agent_positions, k=0, q=10)
                        # secondary update to the env, but it happens first
                        update_enviroment(time_series, eviction[0], eviction[1], location[0], location[1], agent_positions, agent)
                        break
                time_series[-1][location[0]][location[1]] = 3
                
        epoch_to_scores[0].append(rate_performance_policy_3(time_series[-1], hotspot_locations, policy_parameters[3]))  # note initial score
        
        for epoch in range(epochs):
            random.shuffle(agents)
            for agent in agents:
                i, j = agent_positions[agent]
                # square does not contain an agent, ignore
                if time_series[-1][i][j] == 0:
                    continue

                # TODO: UPDATE THESE ACCORDING TO USAGE: if stabilized format for functions, can just make a single call
                # move agent if not happy
                if relocation_policy != 3:
                    happiness = happiness_function(time_series[-1], (i,j), time_series[-1][i][j], k)
                else:
                    happiness = happiness_function(L, hotspot_locations, (i,j), policy_parameters[3])
                
                if happiness != 1:
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
                        if time_series[-1][i][j] !=3:
                            new_i, new_j = policies[3](time_series=time_series, agent=agent, agent_positions=agent_positions,
                                                    desired_value = time_series[-1][i][j],
                                                    heat_map=heat_map, r_relocate_g = policy_parameters[2])
                    elif relocation_policy == 4:
                        policies[4]()

                    # move to new position and adjust visual as well
                    update_enviroment(time_series, new_i, new_j, i, j, agent_positions, agent)
                    

            # at end of each epoch, score the performance
            epoch_to_scores[epoch + 1].append(rate_performance_policy_3(time_series[-1], hotspot_locations, policy_parameters[3]))

        # if display_flag:
        #     display_2D_automata(time_series)
            
    save_start_and_end(time_series, policy_parameters[2])
    return epoch_to_scores


def plot_averages_with_errorbars(averages, std_deviations, labels, filename,
                                 title='Time Series with Averages and Standard Deviation Bars'):
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
    plt.title(title)
    plt.xlabel('Time Points')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'HW2/images/{filename}')
    plt.show()

def plot_averages_with_errorbars_dict(averages, std_deviations, labels, filename,
                                 title='Time Series with Averages and Standard Deviation Bars'):
    colors = ['b','g','r','c','m','y','k']
    # Plotting
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(averages[0]))  # x-values for the data points

    # Plot error bars for A's and B's
    for i, (avg, std) in enumerate(zip(averages, std_deviations)):
        avg = np.array(avg)
        std = np.array(std)
        plt.errorbar(x, avg[:, 0], yerr=std[:, 0], fmt=f'.-{colors[i]}', label=labels[i])
        plt.errorbar(x, avg[:, 1], yerr=std[:, 1], fmt=f'o--{colors[i]}')
        

    # Customize the plot
    plt.title(title)
    plt.xlabel('Time Points')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'HW2/images/{filename}')
    # plt.show()

policies = {0: random_move, 1: social_network_recommendation, 2: rachael_move, 3: connor_move, 4: josh_move}
results1 = simulate_automata(L=40, alpha=0.8, k=6, epochs=20, trials=10, relocation_policy=3, policy_parameters=[2,6,10,10], display_flag=False, happiness_function=get_happyness_connor_move)
results2 = simulate_automata(L=40, alpha=0.8, k=6, epochs=20, trials=10, relocation_policy=3, policy_parameters=[2,6,1,10], display_flag=False, happiness_function=get_happyness_connor_move)
results3 = simulate_automata(L=40, alpha=0.8, k=6, epochs=20, trials=10, relocation_policy=3, policy_parameters=[2,6,.1,10], display_flag=False, happiness_function=get_happyness_connor_move)
results4 = simulate_automata(L=40, alpha=0.8, k=6, epochs=20, trials=10, relocation_policy=3, policy_parameters=[2,6,0.01,10], display_flag=False, happiness_function=get_happyness_connor_move)
results5 = simulate_automata(L=40, alpha=0.8, k=6, epochs=20, trials=10, relocation_policy=3, policy_parameters=[2,6,0.001,10], display_flag=False, happiness_function=get_happyness_connor_move)

averages1= [
    [np.average(results1[epoch_], axis=0) for epoch_ in results1],
    [np.average(results2[epoch_], axis=0) for epoch_ in results2],
    [np.average(results3[epoch_], axis=0) for epoch_ in results3],
    [np.average(results4[epoch_], axis=0) for epoch_ in results4],
    [np.average(results5[epoch_], axis=0) for epoch_ in results5],
]

std_deviations1 = [
    [np.std(results1[epoch_], axis=0) for epoch_ in results1],
    [np.std(results2[epoch_], axis=0) for epoch_ in results2],
    [np.std(results3[epoch_], axis=0) for epoch_ in results3],
    [np.std(results4[epoch_], axis=0) for epoch_ in results4],
    [np.std(results5[epoch_], axis=0) for epoch_ in results5],
]

labels1 = ['k = 10', 'k = 1', 'k = 0.1', 'k = 0.01', 'k = 0.001']
plot_averages_with_errorbars_dict(averages1, std_deviations1, labels1, "policies03.png")

# test random - control case
# results_base = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=0, policy_parameters=[100], display_flag=True)

# test social
# results_n5p3 = simulate_automata(L=40, alpha=.9, k=3, epochs=1, trials=20, relocation_policy=1, policy_parameters=[5, 3, 100], display_flag=True)
# results_n5p5 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[5, 5, 100], display_flag=False)
# results_n10p3 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 3, 100], display_flag=False)
# results_n10p5 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[10, 5, 100], display_flag=False)
# results_n20p3 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[20, 3, 100], display_flag=False)
# results_n20p5 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=1, policy_parameters=[20, 5, 100], display_flag=False)

# averages1= [
#     [np.average(results_base[row]) for row in results_base],
#     [np.average(results_n5p3[row]) for row in results_n5p3],
#     [np.average(results_n5p5[row]) for row in results_n5p5],
#     [np.average(results_n10p3[row]) for row in results_n10p3],
#     [np.average(results_n10p5[row]) for row in results_n10p5],
#     [np.average(results_n20p3[row]) for row in results_n20p3],
#     [np.average(results_n20p5[row]) for row in results_n20p5],
# ]

# std_deviations1 = [
#     [np.std(results_base[row]) for row in results_base],
#     [np.std(results_n5p3[row]) for row in results_n5p3],
#     [np.std(results_n5p5[row]) for row in results_n5p5],
#     [np.std(results_n10p3[row]) for row in results_n10p3],
#     [np.std(results_n10p5[row]) for row in results_n10p5],
#     [np.std(results_n20p3[row]) for row in results_n20p3],
#     [np.std(results_n20p5[row]) for row in results_n20p5],
# ]

# labels1 = ['Random Move with q=100', 'Friend Recommendation with n=5, p=3','Friend Recommendation with n=5, p=5',
#           'Friend Recommendation with n=10, p=3','Friend Recommendation with n=10, p=5',
#           'Friend Recommendation with n=20, p=3','Friend Recommendation with n=20, p=5',]

# plot_averages_with_errorbars(averages1, std_deviations1, labels1, "policies01.png")


# # test rachael policy
# results_w5b10 = simulate_automata(L=40, alpha=.8, k=2, epochs=20, trials=1, relocation_policy=2, policy_parameters=[5, .1, 100], display_flag=True)
# results_w10b10 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=2, policy_parameters=[10, .1, 100], display_flag=False)
# results_w20b10 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=2, policy_parameters=[20, .1, 100], display_flag=False)
# results_w5b20 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=2, policy_parameters=[5, .2, 100], display_flag=False)
# results_w10b20 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=2, policy_parameters=[10, .2, 100], display_flag=False)
# results_w20b20 = simulate_automata(L=40, alpha=.9, k=3, epochs=20, trials=20, relocation_policy=2, policy_parameters=[20, .2, 100], display_flag=False)

# averages2= [
#     [np.average(results_base[row]) for row in results_base],
#     [np.average(results_w5b10[row]) for row in results_w5b10],
#     [np.average(results_w10b10[row]) for row in results_w10b10],
#     [np.average(results_w20b10[row]) for row in results_w20b10],
#     [np.average(results_w5b20[row]) for row in results_w5b20],
#     [np.average(results_w10b20[row]) for row in results_w10b20],
#     [np.average(results_w20b20[row]) for row in results_w20b20],
# ]

# std_deviations2 = [
#     [np.std(results_base[row]) for row in results_base],
#     [np.std(results_w5b10[row]) for row in results_w5b10],
#     [np.std(results_w10b10[row]) for row in results_w10b10],
#     [np.std(results_w20b10[row]) for row in results_w20b10],
#     [np.std(results_w5b20[row]) for row in results_w5b20],
#     [np.std(results_w10b20[row]) for row in results_w10b20],
#     [np.std(results_w20b20[row]) for row in results_w20b20],
# ]

# labels2 = ['Random Move with q=100', 'Neighborhood Search with w=5, beta=10','Neighborhood Search with w=10, beta=10',
#            'Neighborhood Search with w=20, beta=10', 'Neighborhood Search with w=5, beta=20',
#            'Neighborhood Search with w=10, beta=20', 'Neighborhood Search with w=20, beta=20',]

# plot_averages_with_errorbars(averages2, std_deviations2, labels2, "policies02.png")
