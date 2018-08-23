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
def plotData(features, labels):

    reals = []
    fakes = []

    for i in range(len(features)):
        if (labels[i] > 0.5):
            reals.append(features[i])
        else:
            fakes.append(features[i])

    xReal = [reals[i][0] for i in range(len(reals))]
    yReal = [reals[i][1] for i in range(len(reals))]
    xFake = [fakes[i][0] for i in range(len(fakes))]
    yFake = [fakes[i][1] for i in range(len(fakes))]
    
    plt.scatter(xReal, yReal, color = 'blue')
    plt.scatter(xFake, yFake, color = 'orange')
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

# Returns True if (sink -> source)
def isSymmetric(source, sink, sourceDict):
    return 1.0 if (source in sourceDict.get(sink, [])) else 0.0

################################################################################

# (source -> sink) is likely if (source -> mid) && (mid -> sink)
def isTransitive(source, sink, sourceDict):

    count = 0
    total = 0

    for mid in sourceDict.get(source, []):
        if (sink in sourceDict.get(mid, [])):
            count += 1
        total += 1

    try:
        return count / total
    except:
        return 0.0

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

# Returns source's similarity to people who follow sink
def sourceSimilarity(source, sink, sourceDict, sinkDict):

    averageSimilarity = 0.0
    following = sourceDict.get(source, [])
    a = len(following)

    for follower in sinkDict.get(sink, []):
        b = len(sourceDict.get(follower, []))
        union = len(list(set(following + sourceDict.get(follower, []))))
        intersect = a + b - union
        similarity = intersect / union
        averageSimilarity += similarity

    try:
        averageSimilarity /= len(sinkDict.get(sink, []))
        return averageSimilarity
    except:
        return 0.0

################################################################################

# Returns sink's similarity to people that source is following
def sinkSimilarity(source, sink, sourceDict, sinkDict):

    averageSimilarity = 0.0
    followers = sinkDict.get(sink, [])
    a = len(followers)

    for following in sourceDict.get(source, []):
        b = len(sinkDict.get(following, []))
        union = len(list(set(followers + sinkDict.get(following, []))))
        intersect = a + b - union
        similarity = intersect / union
        averageSimilarity += similarity

    try:
        averageSimilarity /= len(sourceDict.get(source, []))
        return averageSimilarity
    except:
        return 0.0

################################################################################

# Converts the given x data to our features
def processFeatures(x, sourceDict, sinkDict):

    newX = []

    for (source, sink) in x:
        # f1 = isSymmetric(source, sink, sourceDict)
        # f2 = isTransitive(source, sink, sourceDict)
        f3 = sourceSimilarity(source, sink, sourceDict, sinkDict)
        f4 = sinkSimilarity(source, sink, sourceDict, sinkDict)
        newX.append((f3, f4))

    return newX

################################################################################

totalStart = timer()

start = timer()
sourceDict, sinkDict, xTrain, yTrain, xDev, yDev = \
            processTrainingFile(TRAIN_FILE)
end = timer()
print("Time taken to process training data: {:.2f} secs".format(end - start))
# print("Number of training instances: {}".format(len(xTrain)))
# print("Number of development instances: {}".format(len(xDev)))

start = timer()
xTest = processTestFile(TEST_FILE)
end = timer()
print("Time taken to process test data: {:.2f} secs".format(end - start))
# print("Number of test instances: {}".format(len(xTest)))

start = timer()
xTrain = processFeatures(xTrain, sourceDict, sinkDict)
xDev = processFeatures(xDev, sourceDict, sinkDict)
xTest = processFeatures(xTest, sourceDict, sinkDict)
end = timer()
print("Time taken to process edges: {:.2f} secs".format(end - start))

xTrain, yTrain = shuffleLists(xTrain, yTrain)
xDev, yDev = shuffleLists(xDev, yDev)

predictions = runNN(xTrain, yTrain, xDev, hidden_layers = [5])

print("First 10 predictions:")
print(predictions[:10])

printAccuracy(yDev, predictions)

'''
evaluate(yDev, predictions)
writeToFile(predictions)
'''

# plotData(xDev, yDev)

totalEnd = timer()
print("Total elapsed time: {:.2f} secs".format(totalEnd - totalStart))

################################################################################
