import numpy as np

def total_dissimilarity(x):
    # Fo
    return np.var(x)*np.shape(x)[0]

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

def dist_matrix(set1, set2):
    set1Len = np.shape(set1)[0]
    set2Len = np.shape(set2)[0]
    distMatrix = np.zeros([set1Len, set2Len])
    for i in range(set1Len):
        for j in range(set2Len):
            if i != j:
                argument = np.reshape(set1[i]-set2[j], 1)
                distMatrix[i,j] = np.linalg.norm(argument, ord=1)
    return distMatrix

def find_nearest_neighbor(set1, set2):
    distMatrix = dist_matrix(set1, set2)
    mask = (distMatrix != 0)
    return np.min(distMatrix[mask])

def find_farthest_neighbor(set1, set2):
    distMatrix = dist_matrix(set1, set2)
    mask = (distMatrix != 0)
    return np.max(distMatrix[mask])

def baricenter_distance(set1, set2):
    return np.abs(np.mean(set1) - np.mean(set2))

def delta_dispersion(set1, set2):
    set1Len = np.shape(set1)[0]
    set2Len = np.shape(set2)[0]

    mean1 = np.mean(set1)
    mean2 = np.mean(set2)

    arg1 = np.abs(mean1 + mean2)**2
    return arg1*(set1Len*set2Len)/(set1Len+set2Len)

# def delta_dispersion(set1, set2):
#     '''
#       Supposedly equivalent formula:
#       Delta F_ij = F_ij - (F_i + F_j)
#     '''
#     F_joint = intraclass_dispersion(np.append(set1, set2))
#     F1 = intraclass_dispersion(set1)
#     F2 = intraclass_dispersion(set2)
#     return F_joint - (F1 + F2)
