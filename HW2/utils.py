import tkinter as tk
import time
import random


def display_2D_automata(automata_time_series):
    root = tk.Tk()
    cell_size = 20
    canvas = tk.Canvas(root, width=cell_size*len(automata_time_series[0]), height=cell_size*len(automata_time_series[0]))
    canvas.pack()

    for sublist in automata_time_series:
        display_colors(canvas, sublist, cell_size)
        root.update()
        time.sleep(1)  # Adjust the sleep duration as needed

    root.mainloop()


def display_colors(canvas, sublist, cell_size):
    canvas.delete("all")
    color_map = {0: "white", 1: "red", 2: "blue"}
    for row_index, row in enumerate(sublist):
        for col_index, value in enumerate(row):
            canvas.create_rectangle(
                col_index * cell_size, row_index * cell_size,
                (col_index + 1) * cell_size, (row_index + 1) * cell_size,
                fill=color_map[value]
            )


def get_x_from_px(x_alphabet, probabilities):
    limits = [sum(probabilities[0:i+1]) for i in range(len(probabilities))]
    x = random.random()
    for val, limit in zip(x_alphabet, limits):
        if x < limit:
            return val
