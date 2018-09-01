from math import sqrt, log
import numpy as np
import pandas as pd
from Constants import *
from FileProcessor import processTrainingFile, processTestFile, \
                          processFeatureFile


# Saves a given list of features to a file
def saveFeatureFile(filename, x, y = None):

    # Initialise header row
    # Skip the max similarity, 
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
# Input: a list of pairwise (source, sink) + source/sink dictionary
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
        
        sourceFeats = sourceSimilarity(source, sink, sourceDict, sinkDict)
        sinkFeats = sinkSimilarity(source, sink, sourceDict, sinkDict)

        features = tuple(sourceFeats + sinkFeats)
        #assert (len(features) == FEATURES)
        
        newX.append(features)

        if (verbose):
            count += 1
            end = timer()
            print("Completed {} / {} ({:.2f} secs)"
                  .format(count, total, end - start))

    return newX

################################################################################

# Calcualte the source's similarity to people who follow sink
# Output: [s1mean, s1max, s2mean,.....]
def sourceSimilarity(source, sink, sourceDict, sinkDict):

    # #Jiaan to Yupei: this part is redundant, no node or edge need to be removed
    # #remove the edge from Dictionary if exist
    
    # try:
    #     sourceDict[source].remove(sink)
    #     sinkDict[sink].remove(source)
    #     label = 1
    # except:
    #     label = 0


    following = sourceDict.get(source, [])
    # Dataframe: columns: different similarity metrics; rows: pairs of followers/following
    similarities = pd.DataFrame(columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7'])
    columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7']

    '''
    Get list of similarities by comparing source's
    following states to those who follow sink
    '''
    print(len(sinkDict.get(sink, [])))
    for i,follower in enumerate(sinkDict.get(sink, [])):
        neighbourFollowing = sourceDict.get(follower, [])
        
        # write a row to dataframe
        similarities.loc[i] = \
            calcualteSimilarity(neighbourFollowing, following)
        #print(similarities.loc[i])


    features = []
    # Calculate mean / max from the dataframe
    for col in columns:
        features.append(similarities[col].mean())
        features.append(similarities[col].max())
    

    # features = []
    # for column in similarities.columns:
    #     mean_, max_ = calculateFeatures(similarities[column])
    #     features.append(mean_)
    #     features.append(max_)

    # #put the removed edge back into the Dict
    # if label == 1:
    #     sourceDict[source].append(sink)
    #     sinkDict[sink].append(source)

    return features
    
################################################################################

# Calculate sink's similarity to people that source is following
# Output: [s1mean, s1max, s2mean,.....]
def sinkSimilarity(source, sink, sourceDict, sinkDict):
    
    # #remove the edge from Dictionary if exist
    # try:
    #     sourceDict[source].remove(sink)
    #     sinkDict[sink].remove(source)
    #     label = 1
    # except:
    #     label = 0
    

    followers = sinkDict.get(sink, [])
    similarities = pd.DataFrame(columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7'])
    columns = ['s1', 's2', 's3', 's4', 's5', 's6', 's7']
    
    '''
    Get list of similarities by comparing sink's
    profile to the profiles source is following
    '''
    for i,following in enumerate(sourceDict.get(source, [])):
        neighbourFollowers = sinkDict.get(following, [])


        similarities.loc[i] = \
            calcualteSimilarity(neighbourFollowers, followers)

    features = []
    # Calculate column mean / max from the dataframe
    for col in columns:
        features.append(similarities[col].mean())
        features.append(similarities[col].max())
 
 
    # #output table
    # features = []ImportError
    # for column in similarities.columns:
    #     mean_, max_ = calculateFeatures(similarities[column])
    #     features.append(mean_)
    #     features.append(max_)

    # #put the removed edge back into the Dict
    # if label == 1:
    #     sourceDict[source].append(sink)
    #     sinkDict[sink].append(source)

    return features

#############################################################################

# similarity methods indexed corresponding to https://arxiv.org/pdf/0901.0553.pdf
# Input: two lists of followers / following
# Output: a list of similarities
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
                        processTrainingFile(TRAIN_FILE, verbose)
xTest = processTestFile(TEST_FILE)

# Creates new data files to use in place of the given ones
# Convert files to our features

start = timer()
xTrain = processFeatures(xTrain, sourceDict, sinkDict, verbose)
saveFeatureFile("training-features.txt", xTrain, yTrain)
xDev = processFeatures(xDev, sourceDict, sinkDict, verbose)
saveFeatureFile("development-features.txt", xDev, yDev)
xTest = processFeatures(xTest, sourceDict, sinkDict, verbose)
saveFeatureFile("test-features.txt", xTest)
end = timer()

if (verbose):
    print("Time taken to process features: {:.2f} secs"
              .format(end - start))

print("Feature files created successfully.")

################################################################################
