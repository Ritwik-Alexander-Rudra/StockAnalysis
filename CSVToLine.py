import matplotlib.pyplot as plt
import csv
import matplotlib.dates as mdates

csvfile = "Data/stock_dfs/AAPL.csv"
csvValues = []

def init(csvfile):
    with open(csvfile, "r") as f:
        csvString = f.read()
        csvValues = csvString.split(",")
        print(csvValues)

    #Deleting Column Names
    for i in range(7):
        del csvValues[i]
    


# csv file columns: date, open, high, low, close, Adj Close, Volume
# Gets column values at given index
def getColValues(index, numcol, isString):
    values = []
    for i in range(len(csvValues)):
        if i % numcol == index:
            if isString:
                values.append(csvValues[i])
            else:
                values.append(int(csvValues[i]))
    return values

def plotLineGraph(csvfile, y, numcol):
    init(csvfile)
    y_val = getColValues(y, numcol, False)
    print(y_val)
    ## X is alWays Date
    x_val = getColValues(0, numcol, True)
    x = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in x_val]
    plt.plot(x,y_val)
    plt.show()

plotLineGraph(csvfile, 4, 7)

##
##    with open(file, "r") as csvfile:
##        plots = csv.reader(csvfile)
##        for row in plots:
##            x.append(row_number)
##            y.append(float(row[0]))
##            row_number += 1
##            print("Successfully Imported and Plotted Row: " + str(row_number))






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
