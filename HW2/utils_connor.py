import matplotlib.pyplot as plt
import time
import random


# def display_2D_automata(automata_time_series):
#     """
#     This function displays the time series of the 2D automata, expecting data in a format of a list of rows of columns
#     :param automata_time_series: list of snapshots of automata at each time
#     :return: two tk.Image objects for the start of the series and end of the series
#     """
#     custom_cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green', 'black'])

#     # Adjust the figsize parameter to set the size of the figure
#     fig, ax = plt.subplots(figsize=(8, 6))

#     def update_plot(time_step):
#         ax.clear()
#         ax.imshow(automata_time_series[time_step], cmap=custom_cmap, vmin=0, vmax=3)
#         ax.set_title(f'Time Step: {time_step}')
#         plt.pause(0.005)

#         if time_step == 5:
#             plt.savefig(f'HW2/images/initial{int(time.time())}.png')
#         elif time_step == len(automata_time_series) - 5:
#             plt.savefig(f'HW2/images/final{int(time.time())}.png')

#     for t in range(len(automata_time_series)):
#         update_plot(t)

#     plt.show()

def save_start_and_end(automata_time_series, k):
    custom_cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green', 'black'])
    # Adjust the figsize parameter to set the size of the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    def save_plot(time_step, k, name):
        ax.clear()
        ax.imshow(automata_time_series[time_step], cmap=custom_cmap, vmin=0, vmax=3)
        ax.set_title(f'{name} Agent Positions Where k = {k}')

        plt.savefig(f'HW2/images/{name}{int(time.time())}.png')
            
    save_plot(5, k, 'Initial')
    save_plot(-2, k, 'Final')

def get_x_from_px(x_alphabet, probabilities):
    """
    Get a realization of x from the given alphabet and probabilities for each value
    :param x_alphabet: values to be selected from ie alphabet of X
    :param probabilities: parallel list of probabilities P_X for each value
    :return: realization of a single selection of x
    """
    limits = [sum(probabilities[0:i+1]) for i in range(len(probabilities))]
    x = random.random()
    for val, limit in zip(x_alphabet, limits):
        if x < limit:
            return val
