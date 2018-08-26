from Constants import *

################################################################################

# Makes very confident predictions based on linearly separating the data
def splitClassifier(filename):

    file = open(filename, 'r')
    lines = file.readlines()[1:]
    x = []

    # Read file into a list of tuples
    for line in lines:
        line = line.strip("\n")
        line = line.split(",")
        f1 = float(line[1])
        f2 = float(line[2])
        x.append((f1, f2))

    split = findSplit(x)
    predictions = []
    
    for (f1, f2) in x:
        if (f1 + f2 > split):
            predictions.append(0.99)
        else:
            predictions.append(0.01)

    return predictions

################################################################################

# Returns a sum that splits the data in half
def findSplit(xTest):
    
    split = -2.0

    # Possible infinite loop is bad...
    while True:
        above = 0
        below = 0
        for (f1, f2) in xTest:
            if (f1 + f2 > split):
                above += 1
            else:
                below += 1

        if abs(above - below) < 5:
            return split
        else:
            split += 0.0001

################################################################################
