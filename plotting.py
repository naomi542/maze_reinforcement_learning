from maze_utils.MM_Maze_Utils import *
import numpy as np
from matplotlib import pyplot as plt

def plot_path(maze, path):
    # plot maze
    ax = PlotMazeCells(maze)

    colours = np.linspace(0, 1, len(path) - 1)
    cmap = plt.get_cmap("jet")

    # plot path arrows
    for i in range(len(path) - 1):
        cur_cell = path[i]
        next_cell = path[i + 1]
        x = maze.xc[cur_cell]
        y = maze.yc[cur_cell]
        dx = 0.8 * (maze.xc[next_cell] - x)
        dy = 0.8 * (maze.yc[next_cell] - y)

        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc=cmap(colours[i]),
                  length_includes_head=True, alpha=0.5, width=0.1)
