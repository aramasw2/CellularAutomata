from auxfunctions import *

import numpy as np
import random
import multiprocessing as mp
from numba import njit

def game_of_life(cell_state, neighborhood):
    #Nan_to_0 to consider padded cells as non-alive in below sum
    alive_neighbors = np.sum(np.nan_to_num(neighborhood, nan=0)) - cell_state  # Exclude the cell itself
    if cell_state == 1:
        return 1 if 2 <= alive_neighbors <= 3 else 0  # Survives if 2-3 neighbors
    else:
        return 1 if alive_neighbors == 3 else 0  # Becomes alive if exactly 3 neighbors

def singlerandomfree_exchange(gridx,ck,height,width):
    pt1 = (np.random.randint(ck,height+ck), np.random.randint(ck,width+ck))
    pt2 = (np.random.randint(ck,height+ck), np.random.randint(ck,width+ck))
    gridx[pt1], gridx[pt2] = gridx[pt2], gridx[pt1]
    
@njit
def singlenearestrandom_exchange(gridx,i,j,m,n):
    
    gridx[i,j], gridx[(m-1+i),(n-1+j)] = gridx[(m-1+i),(n-1+j)], gridx[i,j]

def Conwaysgame_of_life(cell_state,neighbourhood, c):
    """
    Conway's Game of Life rule.

    :param neighbourhood: the current cell's neighbourhood

    :param cell_state: the index of the current cell

    :return: the state of the current cell at the next timestep
    """
    alive_neighbors = np.sum(np.nan_to_num(neighborhood, nan=0)) - cell_state  # Exclude the cell itself
    if cell_state == 1:
        if alive_neighbors < 2:
            return 0  # Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
        if alive_neighbors == 2 or alive_neighbors == 3:
            return 1  # Any live cell with two or three live neighbours lives on to the next generation.
        if alive_neighbors > 3:
            return 0  # Any live cell with more than three live neighbours dies, as if by overpopulation.
    else:
        if alive_neighbors == 3:
            return 1  # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
        else:
            return 0