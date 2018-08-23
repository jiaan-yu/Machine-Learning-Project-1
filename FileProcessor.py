from random import choice as random, shuffle

from Constants import *

################################################################################

# Takes the training file and returns lists of training and test edges
def processTrainingFile(file):

    trainReal, devReal, sourceDict, sinkDict = getEdges(file)
    trainFake, devFake = \
               getFakeEdges(sourceDict, sinkDict, len(trainReal) + len(devReal))

    xTrain = trainReal + trainFake
    yTrain = [REAL for i in range(len(trainReal))] \
           + [FAKE for i in range(len(trainFake))]

    xDev = devReal + devFake
    yDev = [REAL for i in range(len(devReal))] \
         + [FAKE for i in range(len(devFake))]

    return sourceDict, sinkDict, xTrain, yTrain, xDev, yDev

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

    sourceDict = {}
    sinkDict = {}
    
    for (source, sink) in trainReal:
        sourceDict[source] = sourceDict.get(source, [])
        sourceDict[source].append(sink)
        sinkDict[sink] = sinkDict.get(sink, [])
        sinkDict[sink].append(source)
        

    return trainReal, devReal, sourceDict, sinkDict

################################################################################

# Returns a list of N fake edges that do not exist in the training or test data
def getFakeEdges(sourceDict, sinkDict, n):

    fakeEdges = []

    # Make a list of all nodes that appear in the training data
    potentialSources = list(sourceDict.keys())
    potentialSinks = list(sinkDict.keys())

    while (len(fakeEdges) < n):
        source = random(potentialSources)
        sink = random(potentialSinks)

        # Make sure it isn't duplicate nor following itself
        if (source != sink and sink not in sourceDict[source]
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
