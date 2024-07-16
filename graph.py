import matplotlib.pyplot as plt
import numpy as np

def func(x):
    return np.piecewise(x, 
                        [x <= 2, x > 2], 
                        [lambda x: 2*(x-1) + 2, 
                         lambda x: -3*(x-2) + 4])

x = np.linspace(1, 3, 200)  
y = func(x)

plt.plot(x, y, color='blue')

plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Sample graph!')

plt.xlim(1, 3)
plt.ylim(1, 4)
plt.xticks(np.arange(1, 3.5, 0.5))
plt.yticks(np.arange(1, 4.5, 0.5))


plt.show()