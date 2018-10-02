import numpy as np

def total_dissimilarity(x):
    # Fo
    mean = np.mean(x)
    return np.sum(np.abs(x - mean)**2)

def mean_intraclass_dispersion(set):
    # sigma_j
    return total_dissimilarity(set)/np.shape(set)[0]

def intraclass_dispersion(set):
    # F_j
    return total_dissimilarity(set)

def total_intraclass_dispersion(setList):
    # Fin
    Fin = 0
    for set in setList:
        Fin += intraclass_dispersion(set)
    return Fin

def total_interclass_dispersion(setList):
    # Fout
    setList = np.array(setList)
    numSets = np.shape(setList)[0]
    Fout = 0
    # print(setList.shape)
    # print(setList)
    # print(np.mean(setList))

    globalMean = 0
    if (numSets > 1) and (setList.ndim > 0):
        for set in setList:
            globalMean += np.mean(set)
        globalMean /= numSets
    else:
        globalMean = np.mean(setList)

    for set in setList:
        mean = np.mean(set)
        numElements = np.shape(set)[0]

        Fout += numElements*(np.abs(mean - globalMean)**2)
    return Fout
