import numpy as np

import functions

np.set_printoptions(precision=3)

x = np.array([-3.2, -3.0, -0.7, 0.0, 0.8, 2.0, 2.4, 2.9])


# Q1
# Letra a
m0 = np.mean(x)

print("\nx mean: m_0 = ", m0)

Fo = functions.total_dissimilarity(x)

print("\nx total dissimilarity: F_0 = ", Fo)

# Letra b
C1 = np.array([-3.2, -3.0])
C2 = np.array([-0.7, 0.0, 0.8])
C3 = np.array([2.0, 2.4])

setList = np.array([C1, C2, C3])

m1 = np.mean(C1)
m2 = np.mean(C2)
m3 = np.mean(C3)

# print()
# print("Center 1: ", m1)
# print("Center 2: ", m2)
# print("Center 3: ", m3)

F1 = functions.intraclass_dispersion(C1)
F2 = functions.intraclass_dispersion(C2)
F3 = functions.intraclass_dispersion(C3)

print()
print("F1 = ", F1)
print("F2 = ", F2)
print("F3 = ", F3)

# Fin = F1 + F2 + F3
#
# print("\nFin sum = ", Fin)

# Fout = 0
# for set in [C1, C2, C3]:
#     F = len(set)*(np.abs(np.mean(set) - m0)**2)
#     Fout += F
#
# print("\nFout old = ", Fout)

Fin  = functions.total_intraclass_dispersion(setList)
Fout = functions.total_interclass_dispersion(setList)

print("\nFin func = ", Fin)
print("\nFout func= ", Fout)
print("\nFin + Fout:{:.2f} + {:.2f} = {:.2f}".format(Fin, Fout, Fin+Fout))
print("\nFo = ", Fo)

# print("\nF0 = Fin + Fout:\nFin + Fout = {:.2f} + {:.2f} = {:.2f}".format(Fin, Fout, Fin+Fout))


def dist(a, b):
    return np.abs(a - b)

classes = [C1, C2, C3]
numClasses = len(classes)
# d_min  = np.zeros((numClasses, numClasses))
# d_max  = np.zeros((numClasses, numClasses))
# d_mean = np.zeros((numClasses, numClasses))
# for i in range(numClasses):
#     for j in range(numClasses):
#         if (j > i):
#             classA = classes[i]
#             classB = classes[j]
#             distMatrix = np.zeros((len(classA), len(classB)))
#             d_mean[i, j] = np.mean(classA) - np.mean(classB)
#
#             for index in range(len(classA)):
#                 for
#                 distMatrix[]
#
#
# print(d_mean)
