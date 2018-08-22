from random import choice as random, shuffle

from Constants import *

################################################################################

# Takes the training file and returns lists of training and test edges
def processTrainingFile(file):

    trainReal, devReal, edgeDict = getEdges(file)
    trainFake, devFake = getFakeEdges(edgeDict, len(trainReal) + len(devReal))

    xTrain = trainReal + trainFake
    yTrain = [REAL for i in range(len(trainReal))] \
           + [FAKE for i in range(len(trainFake))]

    xDev = devReal + devFake
    yDev = [REAL for i in range(len(devReal))] \
         + [FAKE for i in range(len(devFake))]

    return edgeDict, xTrain, yTrain, xDev, yDev

################################################################################

# Processes a file and returns a list and dictionary detailing the edges
def getEdges(file):

    data = open(file, 'r')
    nodes = []
    edges = []

    for line in data.readlines():
        line = line.split("\t")
        source = int(line[0])
        sinks = line[1:]
        for sink in sinks:
            sink = int(sink)
            edges.append((source, sink))
            nodes.append(sink)
        nodes.append(source)

    print("Number of nodes: {}".format(len(list(set(nodes)))))
    print("Number of edges: {}".format(len(edges)))    

    # Randomise edges and reduce to appropriate sizes
    shuffle(edges)
    edges = edges[:TRAINING_LIMIT + DEV_LIMIT]
    trainReal = edges[:TRAINING_LIMIT]
    devReal = edges[-DEV_LIMIT:]

    edgeDict = {}
    
    for (source, sink) in trainReal:
        edgeDict[source] = edgeDict.get(source, [])
        edgeDict[source].append(sink)

    return trainReal, devReal, edgeDict

################################################################################

# Returns a list of N fake edges that do not exist in the training or test data
def getFakeEdges(edgeDict, n):

    fakeEdges = []

    # Make a list of all nodes that appear in the training data
    potentialSources = list(edgeDict.keys())
    potentialSinks = []
    for key in edgeDict.keys():
        for value in edgeDict[key]:
            potentialSinks.append(value)
        potentialSinks.append(key)
    potentialSinks = list(set(potentialSinks))

    while (len(fakeEdges) < n):
        source = random(potentialSources)
        sink = random(potentialSinks)

        # Make sure it isn't duplicate nor following itself
        if (source != sink and sink not in edgeDict[source]
            and (source, sink) not in fakeEdges):

            fakeEdges.append((source, sink))

    trainFake = fakeEdges[:TRAINING_LIMIT]
    devFake = fakeEdges[-DEV_LIMIT:]

    return trainFake, devFake

################################################################################

# Returns the list of edges defined in the test file
def processTestFile(file):

    data = open(file, 'r')

    xTest = []

    for line in data.readlines():
        line = line.split("\t")

        try:
            source = int(line[1])
            sink = int(line[2])
            xTest.append((source, sink))

        # Header, ignore it
        except:
            continue

    return xTest

################################################################################
