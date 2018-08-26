from random import random, shuffle
from matplotlib import pyplot as plt

from Constants import *
from Evaluation import evaluate, printAccuracy
from W5_Adaptation import runNN
from FeatureProcessor import processFeatures
from SplitClassifier import splitClassifier
from NeighbourClassifier import neighbourClassifier
from FileProcessor import processTrainingFile, processTestFile, \
                          processFeatureFile

################################################################################            

# Returns a random prediction for all test instances
def randomClassifier(x):
    return [random() for i in range(len(x))]

################################################################################

# Plots the given data with the colour representing real and fake
def plotData(features, labels, title = None):

    reals = []
    fakes = []

    for i in range(len(labels)):
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

    if (title is not None):
        plt.title(title)
    
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

# Saves a given list of features to a file
def saveFeatureFile(filename, x, y = None):

    # Initialise header row
    data = "Id"
    for i in range(0, FEATURES):
        data += ",f{}".format(i + 1)

    if (y is None):
        data += "\n"
    else:
        data += ",label\n"

    # Write instances as rows
    for i in range(len(x)):

        data += "{}".format(i)
        for j in range(0, FEATURES):
            data += ",{}".format(x[i][j])

        # Add label to last column if applicable
        if (y is None):
            data += "\n"
        else:
            data += ",{}\n".format(y[i])

    file = open(filename, 'w')
    file.write(data)
    file.close()
    print("File saved successfully.")

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

# Creates new data files to use in place of the given ones
def createFeatureFiles(verbose = False):

    start = timer()
    sourceDict, sinkDict, xTrain, yTrain, xDev, yDev = \
                processTrainingFile(TRAIN_FILE, verbose = verbose)
    end = timer()
    
    if (verbose):
        print("Time taken to process training data: {:.2f} secs"
              .format(end - start))
        print("Number of training instances: {}".format(len(xTrain)))
        print("Number of development instances: {}".format(len(xDev)))

    start = timer()
    xTest = processTestFile(TEST_FILE)
    end = timer()
    
    if (verbose):
        print("Time taken to process test data: {:.2f} secs"
              .format(end - start))
        print("Number of test instances: {}".format(len(xTest)))

    # Convert files to our features
    start = timer()
    xTrain = processFeatures(xTrain, sourceDict, sinkDict, verbose = verbose)
    saveFeatureFile("training-features.txt", xTrain, yTrain)
    xDev = processFeatures(xDev, sourceDict, sinkDict, verbose = verbose)
    saveFeatureFile("development-features.txt", xDev, yDev)
    xTest = processFeatures(xTest, sourceDict, sinkDict, verbose = verbose)
    saveFeatureFile("test-features.txt", xTest)
    end = timer()

    if (verbose):
        print("Time taken to process features: {:.2f} secs"
              .format(end - start))

    print("Feature files created successfully.")

################################################################################

# Ensures no features have values of 0.0
def addE(x):

    for i in range(len(x)):
        x[i] = list(x[i])
        for j in range(FEATURES):
            x[i][j] += 0.001
        x[i] = tuple(x[i])

    return x

################################################################################

# Reduces the dimensionality of x to the two selected features
def reduceFeatures(x, indexes):

    reducedX = []

    for i in range(len(x)):
        features = [x[i][index] for index in indexes]
        reducedX.append(tuple(features))

    return reducedX

################################################################################

totalStart = timer()

xTrain, yTrain = processFeatureFile(TRAINING_FEATURES_FILE)
xTrain, yTrain = shuffleLists(xTrain, yTrain)

xTrain = xTrain[:2 * TRAINING_LIMIT]
yTrain = yTrain[:2 * TRAINING_LIMIT]

xDev, yDev = processFeatureFile(DEVELOPMENT_FEATURES_FILE)
xDev, yDev = shuffleLists(xDev, yDev)

xTest = processFeatureFile(TEST_FEATURES_FILE)

# Ensure no features are 0.0
xTrain = addE(xTrain)
xDev = addE(xDev)
xTest = addE(xTest)

features = [4]
hidden_layers = [2]

xTrain = reduceFeatures(xTrain, features)
xDev = reduceFeatures(xDev, features)
xTest = reduceFeatures(xTest, features)

testing = True

if (testing):
    predictions = runNN(xTrain, yTrain, xTest, hidden_layers = hidden_layers)
else:
    predictions = runNN(xTrain, yTrain, xDev, hidden_layers = hidden_layers)
    evaluate(yDev, predictions)

writeToFile(predictions)

totalEnd = timer()
print("Features: {}".format(features))
print("Hidden Layers: {}".format(hidden_layers))
print("Total elapsed time: {:.2f} secs".format(totalEnd - totalStart))

################################################################################
