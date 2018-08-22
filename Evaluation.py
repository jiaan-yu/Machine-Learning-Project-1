from matplotlib import pyplot as plt

from Constants import *

################################################################################

# Prints a summary of the results
def evaluate(real, predictions):
    printAccuracy(real, predictions)
    AUC = calculateAUC(real, predictions, plotAUC = False)
    print("AUC: {:.2f}".format(AUC))
    print("Mark = {:.1f} / 9".format(9 * max(min(AUC, 0.9) - 0.4, 0.0) / 0.5))

################################################################################

# Prints the accuracy of the predictions, along with average error
def printAccuracy(real, predictions):

    error_sum = 0.0
    correct = 0
    total = 0
    for i in range(len(real)):
        diff = abs(real[i] - predictions[i])
        if (diff < 0.5):
            correct += 1
        total += 1
        error_sum += 1

    # print("Error Sum = {:.2f}".format(error_sum))
    # print("Average Error = {:.2f}".format(error_sum / total))
    print("Accuracy = {:.2f}%".format(100 * correct / total))

################################################################################

# Returns the AUC for the given true answers and predictions
def calculateAUC(real, predictions, plotAUC = False):
    
    fprs, tprs = getRates(real, predictions)

    auc = 0.0
    i = 1
    lastIndex = 0
    
    for i in range(len(tprs)):
        if (tprs[i] != tprs[lastIndex]):
            auc += tprs[lastIndex] * (fprs[lastIndex] - fprs[i])
            lastIndex = i
    auc += tprs[lastIndex] * (1.0 - fprs[lastIndex])

    if (plotAUC):
        plt.plot(fprs, tprs)
        plt.show()
    
    return auc

################################################################################

# Returns the list of true positive and false positive rates for the given data
def getRates(real, predictions, n = 1000):

    step = 1.0 / n
    fprs, tprs = [], []
    
    for i in range(n):

        # Convert to definitive answers based on this level of confidence
        preds = [REAL if predictions[j] > (i * step) else FAKE
                 for j in range(len(predictions))]
        
        TP, FP, FN, TN = getConfusionMatrix(real, preds)
        
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)

        fprs.append(FPR)
        tprs.append(TPR)
        
    return fprs, tprs

################################################################################

# Returns a confusion matrix based on the given true answers and predictions
def getConfusionMatrix(real, predictions):

    TP, FP, FN, TN = 0, 0, 0, 0
    
    for i in range(len(real)):
        if (real[i] == 1 and predictions[i] == 1):
            TP += 1
        elif (real[i] == 0 and predictions[i] == 1):
            FP += 1
        elif (real[i] == 1 and predictions[i] == 0):
            FN += 1
        else:
            TN += 1

################################################################################
    return TP, FP, FN, TN
