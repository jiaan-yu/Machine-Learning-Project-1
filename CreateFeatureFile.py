from math import sqrt, log
import numpy as np
import pandas as pd
from Constants import *
from FileProcessor import processTrainingFile, processTestFile, \
                          processFeatureFile



# Saves a given list of features to a file
def saveFeatureFile(filename, x, y = None):

    # Initialise header row
    data = "Id"
    for i in range(0, len(x[0])):
        data += ",f{}".format(i + 1)

    if (y is None):
        data += "\n"
    else:
        data += ",label\n"

    # Write instances as rows
    for i in range(len(x)):

        data += "{}".format(i)
        for j in range(0, len(x[0])):
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
################################################################################
################################
# Converts the given x data to our features
def processFeatures(x, sourceDict, sinkDict, verbose = False):

    newX = []

    count = 0
    total = len(x)

    start = timer()

    '''
    Features
    f1 - Average similarity between source and those who follow sink
    f2 - Standard deviation in similarity between
            source and those who follow sink
    f3 - Maximum similarity between source and those who follow sink
    f4 - 1.0 / average similarity, which is used in exponential distributions
    
    f5 - Average similarity between sink and those who source follows
    f6 - Standard deviation in similarity between
            sink and those who source follows
    f7 - Maximum similarity between sink and those who source follows
    f8 - 1.0 / average similarity, which is used in exponential distributions
    '''
    for (source, sink) in x:
        
        sourceFeats = sourceSimilarity(source, sink, sourceDict, sinkDict)
        sinkFeats = sinkSimilarity(source, sink, sourceDict, sinkDict)

        features = tuple([f for f in sourceFeats] + [f for f in sinkFeats])
        #assert (len(features) == FEATURES)
        
        newX.append(features)

        if (verbose):
            count += 1
            end = timer()
            print("Completed {} / {} ({:.2f} secs)"
                  .format(count, total, end - start))

    return newX

################################################################################

# Returns True if (sink -> source)
def isSymmetric(source, sink, sourceDict):
    return 1 if (source in sourceDict.get(sink, [])) else -1



################################################################################

# Returns source's similarity to people who follow sink
# return a list of features
def sourceSimilarity(source, sink, sourceDict, sinkDict):

  #remove the edge from Dictionary if exist
    
    try:
        sourceDict[source].remove(sink)
        sinkDict[sink].remove(source)
        label = 1
    except:
        label = 0
    
    following = sourceDict.get(source, [])
    similarities = pd.DataFrame(columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7'])

    '''
    Get list of similarities by comparing source's
    following tastes to those who follow sink
    '''
    for follower in sinkDict.get(sink, []):
        neighbourFollowing = sourceDict.get(follower, [])
        
        similarities.loc[len(similarities)] = \
            calcualteSimilarity(neighbourFollowing, following)
    
    #output table
    features = []
    for column in similarities.columns:
        mean_, max_ = calculateFeatures(similarities[column])
        features.append(mean_)
        features.append(max_)

    #put the removed edge back into the Dict
    if label == 1:
        sourceDict[source].append(sink)
        sinkDict[sink].append(source)
    return features
    
################################################################################

# Returns sink's similarity to people that source is following
def sinkSimilarity(source, sink, sourceDict, sinkDict):
    
    #remove the edge from Dictionary if exist
    try:
        sourceDict[source].remove(sink)
        sinkDict[sink].remove(source)
        label = 1
    except:
        label = 0
    

    similarities = pd.DataFrame(columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7'])
    followers = sinkDict.get(sink, [])
    '''
    Get list of similarities by comparing sink's
    profile to the profiles source is following
    '''
    for following in sourceDict.get(source, []):
        neighbourFollowers = sinkDict.get(following, [])
        similarities.loc[len(similarities)] = \
            calcualteSimilarity(neighbourFollowers, followers)

    #output table
    features = []
    for column in similarities.columns:
        mean_, max_ = calculateFeatures(similarities[column])
        features.append(mean_)
        features.append(max_)

    #put the removed edge back into the Dict
    if label == 1:
        sourceDict[source].append(sink)
        sinkDict[sink].append(source)

    return features

#############################################################################
# Calculates features based on several aspects of node similarities
def calculateFeatures(similarities):

    try:
        meanSimilarity = sum(similarities) / len(similarities)
        maxSimilarity = max(similarities)
        return (meanSimilarity, maxSimilarity)
    except:
        # In case len(similarities) == 0
        return (0, 0)

# similarity methods indexed corresponding to https://arxiv.org/pdf/0901.0553.pdf
def calcualteSimilarity(x, y):
    # x, y should be non-empty set of neighbours
    x = set(x)
    y = set(y)

    try:
        s1 = len(x & y)
        s2 = len(x & y) / sqrt(len(x) * len(y))
        s3 = len(x & y) / len(x | y)
        s4 = 2 * len(x & y) / (len(x) + len(y))
        s5 = len(x & y) / min(len(x), len(y))
        s6 = len(x & y) / max(len(x), len(y))
        s7 = len(x & y) / (len(x) * len(y))
        return s1, s2 ,s3, s4, s5, s6, s7
    except: # len(x)*len(y) == 0
        return (0, 0, 0, 0, 0, 0, 0)


################################################################################
verbose = True

sourceDict, sinkDict, xTrain, yTrain, xDev, yDev =\
                        processTrainingFile(TRAIN_FILE, verbose = True)

xTest = processTestFile(TEST_FILE)

    # Creates new data files to use in place of the given ones
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

################################################
