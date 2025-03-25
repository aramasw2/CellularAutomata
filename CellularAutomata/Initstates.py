import numpy as np
import auxfunctions as auxf

def random_state(grid):
    """Fill the given grid with random 0s and 1s."""
    grid[:] = np.random.choice([0, 1], size=grid.shape)
    return grid

def partitioned_state(grid):
    """Create left side full of 0s and right full of 1s"""
    height, width = grid.shape
    grid[0:height,int(width/2):width] = 1
    return grid

def all_zeros(grid):
    """Set all values in the grid to 0."""
    grid[:] = 0
    return grid

def all_ones(grid):
    """Set all values in the grid to 1."""
    grid[:] = 1
    return grid

def checkerboard(grid):
    """Fill the grid with a checkerboard pattern."""
    grid[:] = 0
    grid[::2, ::2] = 1
    grid[1::2, 1::2] = 1
    return grid

def single_center(grid):
    """Set a single active cell in the center of the grid."""
    grid[:] = 0
    center_x, center_y = grid.shape[1] // 2, grid.shape[0] // 2
    grid[center_y, center_x] = 1
    return grid

def single_random(grid):
    """Set a single active cell randomly in the grid."""
    m, n = auxf.randomchoicearray(grid)
    grid[m,n] = 1
    return grid
