## This is course material for Introduction to Python Scientific Programming
## Example code: multiple_minima.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

fig = plt.figure()
ax = plt.axes()
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Function y = 1/4 x^4 + 1/3 x^3 - x^2 - 2 
def func(x):
    return 1/4*x**4 + 1/3*x**3 - x**2 - 2

def grad(x):
    return x**3 + x**2 - 2*x

x = np.arange(-5, 5, 0.1)
y = func(x)

plt.plot(x, y, 'r-', linewidth = 3)
line = None
def onclick(event):
    global line 

    if not line == None:
        line.remove()

    epsilon = 0.001
    learn_rate = 0.1
    delta = np.inf
    xlist = [event.xdata]  
    ylist = [func(event.xdata)]
    while delta > epsilon:
        x_next = xlist[-1] - learn_rate*grad(xlist[-1])
        delta = abs(xlist[-1] - x_next)
        xlist.append(x_next)
        ylist.append(func(x_next))

    line, = plt.plot(xlist, ylist, 'bo-')

cursor = Cursor(ax, horizOn = True, vertOn = True, color = 'green')
fig.canvas.mpl_connect('button_press_event',onclick)

plt.show()