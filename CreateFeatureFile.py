from math import sqrt, log
import numpy as np
import pandas as pd
from Constants import *
from FileProcessor import processTrainingFile, processTestFile, \
                          processFeatureFile


################################################################################

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

# Features generated based on different similarity metrics
# Input: a list of tuple (source, sink) + source/sink dictionary
# Output: a list which contains features for each pair in tuple (source feats 
# + sink feats)

def processFeatures(x, sourceDict, sinkDict, verbose = False):

    newX = []

    count = 0
    total = len(x)

    start = timer()

    '''
    Features(similarity formula are from https://arxiv.org/pdf/0901.0553.pdf)
    f1 - Average similarity between source and those who follow sink (formula 1)
    f2 - Maximum similarity between source and those who follow sink (formula 1)
    f3 - Average similarity between source and those who follow sink (formula 2)
    f4 - Maximum similarity between source and those who follow sink (formula 2)
    ...
    ...
    f13 - Average similarity between source and those who follow sink (formula 7)
    f14 - Maximum similarity between source and those who follow sink (formula 7)
    f15 - Average similarity between sink and those who source follows (formula 1)
    f16 - Maximum similarity between sink and those who source follows (formula 1)
    ...
    ...
    f27 - Average similarity between sink and those who source follows (formula 7)
    f28 - Maximum similarity between sink and those who source follows (formula 7)
    '''


    for (source, sink) in x:
        features = SourceAndSinkSimilarity(source, sink, sourceDict, sinkDict)
        newX.append(features)

        if (verbose):
            count += 1
            end = timer()
            print("Completed {} / {} ({:.2f} secs)"
                  .format(count, total, end - start))

    return newX

################################################################################

# Calcualte the similarity between source and sink
# Output: a tuple of features for (source, sink)

def SourceAndSinkSimilarity(source, sink, sourceDict, sinkDict):
    
    followings = sourceDict.get(source, []).copy()
    followers = sinkDict.get(sink, []).copy()
    
    # remove the edge if existing
    try:
        followings.remove(sink)
        followers.remove(source)
    except:
        pass

    SourceSimilarities = pd.DataFrame(columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7'])
    SinkSimilarities = pd.DataFrame(columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7'])


    #source similarity
    for i,follower in enumerate(followers):
        neighbourFollowings = sourceDict.get(follower, [])
        SourceSimilarities.loc[i] = \
            calcualteSimilarity(neighbourFollowings, followings)
    
    #Sink Similarities
    for i,following in enumerate(followings):
        neighbourFollowers = sinkDict.get(following, [])
        SinkSimilarities.loc[i] = \
            calcualteSimilarity(neighbourFollowers, followers)
    
    #output mean and max value in each column as features
    features = []
    for col in SourceSimilarities.columns:
        features.append(SourceSimilarities[col].mean())
        features.append(SourceSimilarities[col].max()) 
    for col in SinkSimilarities.columns:
        features.append(SinkSimilarities[col].mean())
        features.append(SinkSimilarities[col].max()) 
    return tuple(features)

################################################################################

# similarity methods indexed corresponding to https://arxiv.org/pdf/0901.0553.pdf
def calcualteSimilarity(x, y):
    # x, y should be non-empty set of neighbours
    x = set(x)
    y = set(y)

    if x and y:
        s1 = len(x & y)
        s2 = len(x & y) / sqrt(len(x) * len(y))
        s3 = len(x & y) / len(x | y)
        s4 = 2 * len(x & y) / (len(x) + len(y))
        s5 = len(x & y) / min(len(x), len(y))
        s6 = len(x & y) / max(len(x), len(y))
        s7 = len(x & y) / (len(x) * len(y))
        return [s1, s2 ,s3, s4, s5, s6, s7]
    return [0, 0, 0, 0, 0, 0, 0]


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
