import rulesets as rs
import Initstates as inst
import entropies as ent

import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import copy as copy
import gzip
import os
import random
import multiprocessing as mp
from numba import njit

import matplotlib
#matplotlib.use('Agg') #Use non-interactive backend to prevent plot during init

class CellularAutomaton:
    def __init__(self, width, height, initst, rule_function, boundary="open"):
        """Initialize all properties and rule function."""
        self.width = width
        self.height = height
        self.initst = initst  # Random initial state       
        self.grid = np.zeros((height,width)) #Zero grid
        self.grid_states = [] # List of gridstates
        #self.rule_function = rule_function
        
        # Convert rule_function from string to actual function reference if needed
        if isinstance(rule_function, str):
            self.rule_function = getattr(rs, rule_function, None)
            if self.rule_function is None:
                raise ValueError(f"Function '{rule_function}' not found in module 'ruleset'.")
        else:
            self.rule_function = rule_function  # If already a function, assign directly
   
        self.boundary = boundary  # "open" or "periodic"
        if self.boundary in ("open" and "periodic"):
            self.ckx = 1
        else:
            self.ckx = 0

    def apply_boundary_conditions(self, grid):
        """Apply boundary conditions (periodic or open)."""
        if self.boundary == "periodic":
            return np.pad(grid, pad_width=self.ckx, mode='wrap')  # Wraps around edges
        elif self.boundary == "open":
            return np.pad(grid, pad_width=self.ckx, mode='constant', constant_values=np.nan)  # NaN-padding
        else:
            return grid

    def initialize(self):
        self.grid = np.zeros((self.height,self.width)) #Reinitialize zero grid
        self.grid_states = []  #Reinitialize grid_states
        
        if isinstance(self.initst, str):
            self.initst = getattr(inst, self.initst, None)
            if self.initst is None:
                raise ValueError(f"Initial state '{initst}' not found in 'Initstates'.")
            self.grid = self.initst(self.grid)  # Call function to generate grid
        else:
            self.grid = self.initst  # Directly use provided initial state array
        '''
        if self.initst == "Partitioned":
            self.grid[0:self.height,int(self.width/2):self.width] = 1
        else:
            self.grid=np.random.choice([0, 1], (self.height, self.width)).astype(np.float32)
        '''
        
        # Set up the figure and axis for animation
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.grid, cmap="gray", interpolation="nearest")
        #matplotlib.use('TkAgg') #Restore interactive backend after fig,ax creation
        
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
        
        
    def update(self, frame):
        """Update the grid based on the rule function."""
        new_grid = np.zeros_like(self.grid)
        padded_grid = self.apply_boundary_conditions(self.grid)
        ck = self.ckx
        
        if self.rule_function == rs.singlenearestrandom_exchange:
            i, j = np.random.randint(ck,self.height+ck), np.random.randint(ck,self.width+ck)
            
            neighborhood = self.ptneighborhood(padded_grid,[i,j],type="nearestnbr")

            m, n = randomchoicearray(neighborhood)
            
            self.rule_function(padded_grid,i,j,m,n) # Update the padded grid
            
            self.grid = padded_grid  # Update the grid
            
            if self.boundary in ("open", "periodic"):
                self.grid=self.grid[ck:self.height+ck,ck:self.width+ck]
        
        if self.rule_function == rs.game_of_life:
            
            for i in range(ck,self.height+ck):
                
                for j in range(ck,self.width+ck):
                    
                    # Extract 3x3 neighborhood Moore neighborhood
                    neighborhood = self.ptneighborhood(padded_grid,[i,j],type="nearestnbr")
                    
                    new_grid[i-ck, j-ck] = self.rule_function(padded_grid[i, j], neighborhood)
                    
            self.grid = new_grid  # Update the grid
            
        if self.rule_function == rs.singlerandomfree_exchange:
            self.rule_function(padded_grid,ck,self.height,self.width)            
            self.grid = padded_grid
            
            if self.boundary in ("open", "periodic"):
                self.grid=self.grid[ck:self.height+ck,ck:self.width+ck]
        
        self.im.set_array(self.grid)  # Update the displayed grid
        
        return [self.im]

    #Defunct run_animation function, combined with run
    def run_animation(self, interval=100, frames=100, save=False, savename="cellular_automaton.mp4"):
        """Run the animation and optionally save it."""
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=frames, interval=interval, blit=True)
        
        if save:
            writer = FFMpegWriter(fps=30, metadata={"artist": "Your Name"}, bitrate=1000)
            ani.save(savename, writer=writer) # Save animation
            
        return ani # Returning the animation object
            
    def run(self, steps=10, animate=False, interval=100, frames=100, save=False, filename="cellular_automaton.mp4"):
        """Run the simulation for a given number of steps."""
        # Store grid states for animation (create a list of grids)
        self.grid_states = [self.grid.copy()]  # Start with the initial state
        
        for _ in range(steps):
            self.update(None)
            self.grid_states.append(self.grid.copy())  # Store the updated grid

        # Define animate function, reusing the stored grid states
        def animate_func(frame):
            self.im.set_array(frame)  # Update the display
            return [self.im]

        if animate:
            # Create the animation. Pass frames from grid_states to animate_func
            # Update image for each frame
            ani = animation.FuncAnimation(
                self.fig, animate_func, frames=grid_states, interval=interval, blit=True)
            #plt.show()
            if save:
                writer = FFMpegWriter(fps=30, metadata={"artist": "Your Name"}, bitrate=1000)
                ani.save(filename, writer=writer) # Save animation
    
            return ani  # Return the animation object for further control if needed
            
    def display(self):
        """Display the automaton grid using matplotlib."""
        plt.imshow(self.grid, cmap='binary')
        plt.show()
    
    def output(self):
        return self.grid # returns the grid object

    def outputstates(self):
        return self.grid_states # returns list of all gridstates

    def current_entropy(self):
        return ent.shannon_entropy(str(self.grid))

    def average_entropy(self):
        return ent.average_cell_entropy(np.array(self.grid_states))

    def average_mutualinformation(self):
        return ent.average_mutual_information(np.array(self.grid_states), temporal_distance=1)
        


    

