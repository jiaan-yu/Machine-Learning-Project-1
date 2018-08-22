from Constants import *

################################################################################

# Classifies the data based on nearest neighbours
def neighbourClassifier(x, edgeDict):

    predictions = []

    for i in range(len(x)):

        source = x[i][0]
        sink = x[i][1]
            
        neighbours = getNeighbours(source, edgeDict, k = 10)
        
        if (len(neighbours) == 0):
            # Guess if there are no neighbours
            pred = 0.5
        else:
            # Get the percentage of neighbours that follow this sink
            pred = sum([1 if sink in edgeDict.get(neighbours[j][0], []) else 0
                         for j in range(len(neighbours))]) / len(neighbours)

        # Min-max values
        if (pred <= 0.2):
            pred = 0.01
        else:
            pred = 0.99
            
        predictions.append(pred)
        
    return predictions

################################################################################

# Returns the k nodes closest to the source node
def getNeighbours(source, edgeDict, k = 10, verbose = False):

    sinks = list(set(edgeDict[source]))
    neighbours = [(0, 0) for i in range(k)]

    done = 1

    start = timer()

    for key in edgeDict.keys():

        # Can't be its own neighbour
        if (key == source):
            continue
        
        neighbourSinks = list(set(edgeDict[key]))
        
        # (A ^ B) = A + B - (A v B)
        union = len(list(set(sinks + neighbourSinks)))
        intersect = len(sinks) + len(neighbourSinks) - union
        match = intersect / union

        neighbours.append((key, match))

        for i in range(len(neighbours) - 1, 0, -1):

            # If they're out of order then swap them
            if (neighbours[i][1] > neighbours[i - 1][1]):
                tmp = neighbours[i]
                neighbours[i] = neighbours[i - 1]
                neighbours[i - 1] = tmp
            else:
                break
        
        neighbours = neighbours[:-1]
        
        if (verbose and done % 1000 == 0):
            print("Iterated through {} of {} keys"
                  .format(done, len(edgeDict.keys())))
        done += 1

        # Time limit so none take too long
        current = timer()
        if (current - start > TIME_LIMIT):
            return []
        
    return neighbours        

################################################################################
