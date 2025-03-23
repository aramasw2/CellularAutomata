import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import copy as copy
import gzip
import os

# 0 is white, 1 is black
state=np.zeros((100,100)) #Intialize 100x100 lattice for lattice gas automata
state[0:100,50:100]=1

#Define swap operation of two positions in an array
def swap(state, n1, n2):
    state[n1], state[n2] = state[n2], state[n1]

#Definite animation step to show evolution of 2D lattice

def update_array():
    global state
    origdims =state.shape
    if not isinstance(state, np.ndarray) or state.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    state0 = state.reshape(-1) #Vectorizes array
    dim = state0.size # State dimensions
    #Pick 2 places randomly for nsweeps
    ind1, ind2 = np.random.randint(0,dim-1), np.random.randint(0,dim-1)

    #Ruleset
    swap(state0,ind1,ind2)
    #swap(state0,0,9999)
    state=state0.reshape(origdims)  
    return state  
   
#Define rule set function for calculating final state after nsweeps
def Reconfigure(state,nsweeps):
    origdims =state.shape
    if not isinstance(state, np.ndarray) or state.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    state0 = state.reshape(-1) #Vectorizes array
    dim = state0.size # State dimensions
    #Pick 2 places randomly for nsweeps
    ind1, ind2 = np.random.randint(0,dim-1,size=nsweeps), np.random.randint(0,dim-1,size=nsweeps)
    #Ruleset
    [swap(state0,ind1[n],ind2[n]) for n in np.arange(0, len(ind1))] # Do all nsweep swap operations
    
    return state0.reshape(origdims)

def compute_entropy(state,nx):
    return len(gzip.compress(Reconfigure(state,nx).tobytes()))


fig, ax = plt.subplots()  # Create a figure and an axes object
im = ax.imshow(state, animated=True, cmap='Greys') # Use imshow to display the matrix


def animate(*args):
    global state
    state = update_array() # update the matrix
    im.set_array(state) # update the image
    return im, # return the updated image

ani = animation.FuncAnimation(fig, animate, frames=1000, interval=20,blit=True)
writer = FFMpegWriter(fps=30, metadata={"artist": "Your Name"}, bitrate=1000)
ani.save("CellularAutomaton.mp4", writer=writer)
plt.close()

state=np.zeros((100,100)) #Intialize 100x100 lattice for lattice gas automata
state[0:100,50:100]=1

nsweep_list=np.arange(0,10000,10)
ArrayEntropic=np.array([compute_entropy(state,nx) for nx in nsweep_list])

plt.plot(nsweep_list, ArrayEntropic)
plt.xlabel("Sweeps")
plt.ylabel("Number of compressed bits")
plt.title("Compressibility of cellular automata dynamics")
plt.savefig('myplot.jpg', format='jpeg', dpi=300)
plt.show()