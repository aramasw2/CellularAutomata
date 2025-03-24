import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import random

class CellularAutomaton:
    def __init__(self, width, height, initst, rule_function, boundary="open"):
        """Initialize all properties and rule function."""
        self.width = width
        self.height = height
        self.initst = initst  # Random initial state       
        self.grid = np.zeros((height,width)) #Zero grid
   
        self.rule_function = rule_function  # Function to apply rules
        self.boundary = boundary  # "open" or "periodic"
        if self.boundary == "open":
            self.ckx = 1
        else:
            self.ckx = 0
    
    def apply_boundary_conditions(self, grid):
        """Apply boundary conditions (periodic or open)."""
        if self.boundary == "periodic":
            return np.pad(grid, pad_width=1, mode='wrap')  # Wraps around edges
        elif self.boundary == "open":
            return np.pad(grid, pad_width=1, mode='constant', constant_values=np.nan)  # NaN-padding
        else:
            return grid

    def initialize(self):
        if self.initst == "Partitioned":
            self.grid[0:self.height,int(self.width/2):self.width] = 1
        else:
            self.grid=np.random.choice([0, 1], (self.height, self.width)).astype(np.float32)
        
    def ptneighborhood(self,grid,pt,type="nearestnbr"):
        ck = self.ckx
        i,j = pt
        if type == "nearestnbr":
            nbd=grid[i-1:i+2, j-1:j+2]
            '''
            #For open BCS, mark padded cells with np.nan values
            if self.boundary == "open" and (i in (ck,self.height+ck) or j in (ck,self.width+ck)):
                if i in (ck,self.height+ck):
                    nbd[i, :] = np.nan
                if j in (ck,self.width+ck):
                    nbd[:, j] = np.nan
            '''
        return nbd         
        
        
    def update(self):
        """Update the grid based on the rule function."""
        new_grid = np.zeros_like(self.grid)
        padded_grid = self.apply_boundary_conditions(self.grid)
        ck = self.ckx
        if self.rule_function == singlenearestrandom_exchange:
            i, j = np.random.randint(ck,self.height-ck-1), np.random.randint(ck,self.width-ck-1)
            neighborhood = self.ptneighborhood(padded_grid,[i,j],type="nearestnbr")
            self.grid = self.rule_function(self.grid,[i,j],neighborhood) # Update the grid
        if self.rule_function == game_of_life:
            for i in range(ck,self.height-ck-1):
                for j in range(ck,self.width-ck-1):
                    # Extract 3x3 neighborhood Moore neighborhood
                    neighborhood = self.ptneighborhood(padded_grid,[i,j],type="nearestnbr")
                    padded_grid[i, j] = self.rule_function(self.grid[i, j], neighborhood)
            self.grid = padded_grid  # Update the grid
        elif self.rule_function == singlerandomfree_exchange:
            self.rule_function(padded_grid,ck,self.height,self.width)            
            self.grid = padded_grid
        if self.boundary in ("periodic", "open"):
            self.grid=self.grid[1:self.height-1,1:self.width-1]
        
    def run(self, steps=10):
        """Run the simulation for a given number of steps."""
        for _ in range(steps):
            self.update()
    def output(self):
        return self.grid
        
    def display(self):
        """Display the automaton grid using matplotlib."""
        plt.imshow(self.grid, cmap='binary')
        plt.show()

def randomchoicearray(array,condition):
    indices = np.where(np.vectorize(condition)(array))
    validinds=list(zip(indices[0], indices[1]))    
    return random.choice(validinds)     

def game_of_life(cell_state, neighborhood):
    #Nan_to_0 to consider padded cells as non-alive in below sum
    alive_neighbors = np.sum(np.nan_to_num(neighborhood, nan=0)) - cell_state  # Exclude the cell itself
    if cell_state == 1:
        return 1 if 2 <= alive_neighbors <= 3 else 0  # Survives if 2-3 neighbors
    else:
        return 1 if alive_neighbors == 3 else 0  # Becomes alive if exactly 3 neighbors

def singlerandomfree_exchange(gridx,ck,height,width):
    pt1 = (np.random.randint(ck,height-ck-1), np.random.randint(ck,width-ck-1))
    pt2 = (np.random.randint(ck,height-ck-1), np.random.randint(ck,width-ck-1))
    gridx[pt1], gridx[pt2] = gridx[pt2], gridx[pt1]

def singlenearestrandom_exchange(gridx,pt,neighborhood):
    i, j = pt
    m, n = randomchoicearray(array,lambda x: x != np.nan)
    gridx[i,j], gridx[(m-1+i),(n-1+j)] = gridx[(m-1+i),(n-1+j)], gridx[i,j]
