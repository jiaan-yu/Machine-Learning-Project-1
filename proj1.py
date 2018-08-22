from random import random, shuffle
from matplotlib import pyplot as plt

from Constants import *
from Evaluation import evaluate, printAccuracy
from W5_Adaptation import runNN
from NeighbourClassifier import neighbourClassifier
from FileProcessor import processTrainingFile, processTestFile

################################################################################            

# Returns a random prediction for all test instances
def randomClassifier(x):
    return [random() for i in range(len(x))]

################################################################################

# Plots the given data with the colour representing real and fake
def plotData(edges):

    half = int(len(edges) / 2)
    x = [edges[i][0] for i in range(len(edges))]
    y = [edges[i][1] for i in range(len(edges))]
    plt.scatter(x[half:], y[half:], color = 'blue')
    plt.scatter(x[:half], y[:half], color = 'orange')
    plt.show()

################################################################################

# Moves the predictions from [0, 1] to [0.01, 0.99]
def moveIn(x):
    return (0.98 * x) + 0.01

################################################################################

# Writes the predictions to a file
def writeToFile(predictions):

    # Write the predictions to .csv format
    data = "Id,Prediction\n"
    for i in range(len(predictions)):
        data += "{},{}\n".format(i + 1, predictions[i])

    filename = saveFile(data)

    if (filename is not None):
        print("File saved as: {}".format(filename))
    else:
        print("File could not be saved.")

################################################################################

# Finds a suitable filename and saves the file
def saveFile(data):

    n = 0
    
    while (n < MAX_FILES):
        
        filename = str(SAVE_FILE + str(n) + ".csv")
        
        try:
            file = open(filename, 'r')
            file.close()
            n += 1
        except FileNotFoundError:
            file = open(filename, 'w')
            file.write(data)
            file.close()
            return filename

    # n exceeded MAX_FILES
    return None

################################################################################

# (source -> sink) is likely if (sink -> source)
def symmetricClassifier(xTest, edgeDict):

    predictions = []
    for (source, sink) in xTest:
        pred = 1.0 if (source in edgeDict.get(sink, [])) else 0
        predictions.append(pred)

    return predictions

################################################################################

# (source -> sink) is likely if (source -> mid) && (mid -> sink)
def transitiveClassifier(xTest, edgeDict):

    predictions = []

    for (source, sink) in xTest:
        pred = sum([1 if sink in edgeDict.get(mid, []) else 0
                    for mid in edgeDict[source]]) / len(edgeDict[source])
        predictions.append(pred)

    return predictions

################################################################################

# Shuffles two lists around (both are shuffled in the same way)
def shuffleLists(x, y):

    newOrder = list(range(len(x)))
    shuffle(newOrder)
    newX = []
    newY = []
    
    for i in range(len(newOrder)):
        num = newOrder[i]
        newX.append(x[num])
        newY.append(y[num])

    return newX, newY

################################################################################

totalStart = timer()

start = timer()
edgeDict, xTrain, yTrain, xDev, yDev = processTrainingFile(TRAIN_FILE)
end = timer()
print("Time taken to process training data: {:.2f} secs".format(end - start))
# print("Number of training instances: {}".format(len(xTrain)))
# print("Number of development instances: {}".format(len(xDev)))

start = timer()
xTest = processTestFile(TEST_FILE)
end = timer()
print("Time taken to process test data: {:.2f} secs".format(end - start))
# print("Number of test instances: {}".format(len(xTest)))

xTrain, yTrain = shuffleLists(xTrain, yTrain)
xDev, yDev = shuffleLists(xDev, yDev)

predictions = runNN(xTrain, yTrain, xDev, hidden_layers = [100, 25])

printAccuracy(yDev, predictions)

'''
evaluate(yDev, predictions)
writeToFile(predictions)
plotData(xDev)
'''

totalEnd = timer()
print("Total elapsed time: {:.2f} secs".format(totalEnd - totalStart))

################################################################################
