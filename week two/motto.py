import os 

path = os.path.dirname(os.path.abspath(__file__))
f = open(path + "/" + "motto.txt", "w")

f.write('Fiat Lux!')
f.close()

with open(path + '/motto.txt', 'a') as f:
    f.write('\n Let there be light!')
