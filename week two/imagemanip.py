import os
from matplotlib import image
from matplotlib import pyplot

path = os.path.dirname(os.path.abspath(__file__))
filename1 = path + '/' + 'lenna.bmp'
filename2 = path + '/' + 'usflag.jpg'
lenna = image.imread(filename1)
flag = image.imread(filename2)
lennamod = lenna.copy()

for width in lennamod:
    for width in range(200):
        for height in range(200):
            lennamod[width][height-255] = flag[width][height]
pyplot.imshow(lennamod)
pyplot.show()


