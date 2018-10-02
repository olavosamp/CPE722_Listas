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
C3 = np.array([2.0, 2.4, 2.9])

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

Fin  = functions.total_intraclass_dispersion(setList)
Fout = functions.total_interclass_dispersion(setList)

print("\nFin func = ", Fin)
print("\nFout func= ", Fout)
print("\nFin + Fout:{:.2f} + {:.2f} = {:.2f}".format(Fin, Fout, Fin+Fout))
print("\nFo = ", Fo)

print("Farthest classes: C1, C3")

print("dissimilarity Nearest Neighbor:    d_min  = ", functions.find_nearest_neighbor(C1, C3))
print("dissimilarity Farthest Neighbor:   d_max  = ", functions.find_farthest_neighbor(C1, C3))
print("dissimilarity Baricenter Distance: d_mean = ", functions.baricenter_distance(C1, C3))

# Letra d
numSets = np.shape(setList)[0]
deltaDispersion = np.zeros((3, 3))
for i in range(numSets):
    for j in range(numSets):
        if i<j:
            deltaDispersion[i,j] = functions.delta_dispersion(setList[i], setList[j])

mask = deltaDispersion !=0
print(deltaDispersion)
F13 = functions.intraclass_dispersion(np.append(C1, C3))
deltaF13 = F13 - (F1 + F3)

print(F13)
print(deltaF13)
print(functions.delta_dispersion(C1, C3))
