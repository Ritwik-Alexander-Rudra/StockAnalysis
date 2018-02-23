import matplotlib.pyplot as plt
import csv

x = []
y = []
row_number = 0

with open("AAPL.csv", "r") as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        x.append(row_number)
        y.append(float(row[0]))
        row_number += 1
        print("Successfully Imported and Plotted Row: " + str(row_number))

plt.plot(x,y, label = "Loaded from File!")

plt.show()




##PART 2
##import numpy as np
##
##x, y = np.loadtxt('example.txt', delimiter=',', unpack=True)
##plt.plot(x,y, label='Loaded from file!')
##
##plt.xlabel('x')
##plt.ylabel('y')
##plt.title('Interesting Graph\nCheck it out')
##plt.legend()
##plt.show()
