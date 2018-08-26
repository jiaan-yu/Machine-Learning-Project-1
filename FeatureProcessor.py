from math import sqrt

from Constants import *

################################################################################

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
        assert (len(features) == FEATURES)
        
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
        return -0.5

################################################################################

# Returns source's similarity to people who follow sink
def sourceSimilarity(source, sink, sourceDict, sinkDict):

    averageSimilarity = 0.0
    maxSimilarity = 0.0
    following = sourceDict.get(source, [])
    a = len(following)

    similarities = []

    '''
    Get list of similarities by comparing source's
    following tastes to those who follow sink
    '''
    for follower in sinkDict.get(sink, []):
        neighbourFollowing = sourceDict.get(follower, [])
        b = len(neighbourFollowing)
        union = len(list(set(following + neighbourFollowing)))
        intersect = a + b - union
        sim = (2 * intersect) / (a + b)
        similarities.append(sim)

    return calculateFeatures(similarities)
    
################################################################################

# Returns sink's similarity to people that source is following
def sinkSimilarity(source, sink, sourceDict, sinkDict):

    similarities = []
    followers = sinkDict.get(sink, [])
    a = len(followers)
    
    '''
    Get list of similarities by comparing sink's
    profile to the profiles source is following
    '''
    for following in sourceDict.get(source, []):
        neighbourFollowers = sinkDict.get(following, [])
        b = len(neighbourFollowers)
        union = len(list(set(followers + neighbourFollowers)))
        intersect = a + b - union
        sim = (2 * intersect) / (a + b)
        similarities.append(sim)

    return calculateFeatures(similarities)

################################################################################

# Calculates features based on several aspects of node similarities
def calculateFeatures(similarities):

    try:
        meanSimilarity = sum(similarities) / len(similarities)
        varSimilarity = sum([(similarities[i] - meanSimilarity)**2
                             for i in range(len(similarities))])
        varSimilarity /= len(similarities)
        stdSimilarity = sqrt(varSimilarity)
        maxSimilarity = max(similarities)
        lamb = 1.0 / meanSimilarity

        return (meanSimilarity, stdSimilarity, maxSimilarity, lamb)
    except:
        # In case len(similarities) == 0
        return (0, 0, 0, 0)

################################################################################
