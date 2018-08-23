from Constants import *

################################################################################

# Converts the given x data to our features
def processFeatures(x, sourceDict, sinkDict, verbose = False):

    newX = []

    count = 0
    total = len(x)

    start = timer()

    for (source, sink) in x:
        # f1 = isSymmetric(source, sink, sourceDict)
        # f2 = isTransitive(source, sink, sourceDict)
        f1, f3 = sourceSimilarity(source, sink, sourceDict, sinkDict)
        f2, f4 = sinkSimilarity(source, sink, sourceDict, sinkDict)
        
        newX.append((f1, f2, f3, f4))

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

    for follower in sinkDict.get(sink, []):
        neighbourFollowing = sourceDict.get(follower, [])
        b = len(neighbourFollowing)
        union = len(list(set(following + neighbourFollowing)))
        intersect = a + b - union
        similarity = intersect / union
        averageSimilarity += similarity
        
        if (similarity > maxSimilarity):
            maxSimilarity = similarity
        

    neighbourCount = len(sinkDict.get(sink, []))

    # Normalise to range [-1, 1]
    if (neighbourCount > 0):
        averageSimilarity /= neighbourCount
        return 2.0 * averageSimilarity - 1.0, 2.0 * maxSimilarity - 1.0
    else:
        return -1.0, -1.0
    
################################################################################

# Returns sink's similarity to people that source is following
def sinkSimilarity(source, sink, sourceDict, sinkDict):

    averageSimilarity = 0.0
    maxSimilarity = 0.0
    followers = sinkDict.get(sink, [])
    a = len(followers)

    for following in sourceDict.get(source, []):
        neighbourFollowers = sinkDict.get(following, [])
        b = len(neighbourFollowers)
        union = len(list(set(followers + neighbourFollowers)))
        intersect = a + b - union
        similarity = intersect / union
        averageSimilarity += similarity

        if (similarity > maxSimilarity):
            maxSimilarity = similarity

    neighbourCount = len(sourceDict.get(source, []))

    # Normalise to range [-1, 1]
    if (neighbourCount > 0):
        averageSimilarity /= neighbourCount
        return 2.0 * averageSimilarity - 1.0, 2.0 * maxSimilarity - 1.0
    else:
        return -1.0, -1.0

################################################################################
