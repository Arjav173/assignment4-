import random
import math
import matplotlib.pyplot as plt

def data():
    mean = 1
    sigma = 1
    size = 1000
    dataset = [0.0] * size
    index = 0
    while index < size:
        u1 = 2 * random.random() - 1
        u2 = 2 * random.random() - 1
        s = u1 * u1 + u2 * u2
        if s <= 0 or s >= 1:
            continue
        k = math.sqrt((-2 * math.log(s)) / s)
        x = u1 * k
        y = u2 * k
        dataset[index] = mean + sigma * x
        index += 1
        if index < size:
            dataset[index] = mean + sigma * y
            index += 1
    return dataset

def mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def matrixMultiplication(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    if cols_A != rows_B:
        raise ValueError("Incompatible matrix dimensions")
    result = [[0.0 for j in range(cols_B)] for i in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

def invert_matrix(matrix):
    n = len(matrix)
    A = [[float(value) for value in row] for row in matrix]
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        pivot = A[i][i]
        if abs(pivot) < 0.0000001:
            swap = -1
            for k in range(i + 1, n):
                if abs(A[k][i]) > 0.0000001:
                    swap = k
                    break
            if swap == -1:
                return None
            A[i], A[swap] = A[swap], A[i]
            I[i], I[swap] = I[swap], I[i]
            pivot = A[i][i]
        for j in range(n):
            A[i][j] /= pivot
            I[i][j] /= pivot
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(n):
                    A[k][j] -= factor * A[i][j]
                    I[k][j] -= factor * I[i][j]
    return I

def covarianceMatrix(X, meanVector):
    N = len(X[0])
    c11 = 0
    c12 = 0
    c21 = 0
    c22 = 0
    for i in range(N):
        x1 = X[0][i] - meanVector[0]
        x2 = X[1][i] - meanVector[1]
        c11 += x1 * x1
        c12 += x1 * x2
        c21 += x2 * x1
        c22 += x2 * x2
    return [[c11 / (N - 1), c12 / (N - 1)], [c21 / (N - 1), c22 / (N - 1)]]

def eigenvalues(C):
    a = C[0][0]
    b = C[0][1]
    c = C[1][0]
    d = C[1][1]
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace * trace - 4 * determinant
    lambda1 = (trace + math.sqrt(discriminant)) / 2
    lambda2 = (trace - math.sqrt(discriminant)) / 2
    return lambda1, lambda2

def eigenvector(C, value):
    a = C[0][0]
    b = C[0][1]
    c = C[1][0]
    d = C[1][1]
    if abs(b) > 0.0000001:
        v1 = b
        v2 = value - a
    elif abs(c) > 0.0000001:
        v1 = value - d
        v2 = c
    else:
        if abs(a - value) < abs(d - value):
            v1 = 1
            v2 = 0
        else:
            v1 = 0
            v2 = 1
    magnitude = math.sqrt(v1 * v1 + v2 * v2)
    return [v1 / magnitude, v2 / magnitude]

def covarianceSquareRoot(C):
    lambda1, lambda2 = eigenvalues(C)
    v1 = eigenvector(C, lambda1)
    v2 = eigenvector(C, lambda2)
    V = [[v1[0], v2[0]], [v1[1], v2[1]]]
    D = [[math.sqrt(lambda1), 0], [0, math.sqrt(lambda2)]]
    return matrixMultiplication(matrixMultiplication(V, D), transpose_matrix(V))

def transformData(X, meanVector, C):
    SigmaHalf = covarianceSquareRoot(C)
    result = [[], []]
    N = len(X[0])
    for i in range(N):
        centered = [[X[0][i] - meanVector[0]], [X[1][i] - meanVector[1]]]
        transformed = matrixMultiplication(SigmaHalf, centered)
        y1 = transformed[0][0] + meanVector[0]
        y2 = transformed[1][0] + meanVector[1]
        result[0].append(y1)
        result[1].append(y2)
    return result

def quadraticForm(x, y, meanVector, C):
    inverse = invert_matrix(C)
    a = x - meanVector[0]
    b = y - meanVector[1]
    vector = [[a], [b]]
    vectorT = transpose_matrix(vector)
    result = matrixMultiplication(matrixMultiplication(vectorT, inverse), vector)
    return result[0][0]

def plotContours(X, meanVector, C, title):
    minX = min(X[0]) - 1
    maxX = max(X[0]) + 1
    minY = min(X[1]) - 1
    maxY = max(X[1]) + 1
    numberOfPoints = 200
    xValues = []
    yValues = []
    for i in range(numberOfPoints):
        xValues.append(minX + (maxX - minX) * i / (numberOfPoints - 1))
        yValues.append(minY + (maxY - minY) * i / (numberOfPoints - 1))
    Z = []
    for y in yValues:
        row = []
        for x in xValues:
            row.append(quadraticForm(x, y, meanVector, C))
        Z.append(row)
    plt.figure(figsize=(8, 8))
    plt.scatter(X[0], X[1], s=10, alpha=0.25)
    levels = [1, 2, 3, 4, 5, 6, 8, 10]
    contours = plt.contour(xValues, yValues, Z, levels=levels)
    plt.clabel(contours, inline=True, fontsize=8)
    lambda1, lambda2 = eigenvalues(C)
    v1 = eigenvector(C, lambda1)
    v2 = eigenvector(C, lambda2)
    k = 4
    axis1 = math.sqrt(k * lambda1)
    axis2 = math.sqrt(k * lambda2)
    plt.plot([meanVector[0] - axis1 * v1[0], meanVector[0] + axis1 * v1[0]], [meanVector[1] - axis1 * v1[1], meanVector[1] + axis1 * v1[1]], linewidth=3)
    plt.plot([meanVector[0] - axis2 * v2[0], meanVector[0] + axis2 * v2[0]], [meanVector[1] - axis2 * v2[1], meanVector[1] + axis2 * v2[1]], linewidth=3)
    plt.scatter(meanVector[0], meanVector[1], marker="x", s=100, linewidths=3)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def printMatrix(M):
    for row in M:
        print(row)

def discriminant(x, y, mu, inverse, determinant):
    dx = x - mu[0]
    dy = y - mu[1]
    vector = [[dx], [dy]]
    vectorT = transpose_matrix(vector)
    distance = matrixMultiplication(matrixMultiplication(vectorT, inverse), vector)[0][0]
    return (-0.5 * distance -0.5 * math.log(determinant))

def generateClass(meanVector, C):
    base = [data(), data()]
    Y = transformData(base, [1, 1], C)
    for i in range(len(Y[0])):
        Y[0][i] += meanVector[0] - 1
        Y[1][i] += meanVector[1] - 1
    return Y
X1 = data()
X2 = data()
X = [X1, X2]
mu = [mean(X1), mean(X2)]

print("QUESTION 1")
print("")
C1 = covarianceMatrix(X, mu)
lambda1, lambda2 = eigenvalues(C1)
v1 = eigenvector(C1, lambda1)
v2 = eigenvector(C1, lambda2)
print("Mean:")
print(mu)
print("Covariance:")
printMatrix(C1)
print("Eigenvalues:")
print(lambda1)
print(lambda2)
print("Eigenvectors:")
print(v1)
print(v2)
plotContours(X, mu, C1, "Question 1")

print("QUESTION 2")
print("")
C2 = [[4, 0], [0, 1]]
Y2 = transformData(X, mu, C2)
mu2 = [mean(Y2[0]), mean(Y2[1])]
estimatedC2 = covarianceMatrix(Y2, mu2)
lambda1, lambda2 = eigenvalues(estimatedC2)
v1 = eigenvector(estimatedC2, lambda1)
v2 = eigenvector(estimatedC2, lambda2)
print("Mean:")
print(mu2)
print("Covariance:")
printMatrix(estimatedC2)
print("Eigenvalues:")
print(lambda1)
print(lambda2)
print("Eigenvectors:")
print(v1)
print(v2)
plotContours(Y2, mu2, estimatedC2, "Question 2")

print("QUESTION 3")
print("")
C3 = [[4, 1.5], [1.5, 2]]
Y3 = transformData(X, mu, C3)
mu3 = [mean(Y3[0]), mean(Y3[1])]
estimatedC3 = covarianceMatrix(Y3, mu3)
lambda1, lambda2 = eigenvalues(estimatedC3)
v1 = eigenvector(estimatedC3, lambda1)
v2 = eigenvector(estimatedC3, lambda2)
print("Mean:")
print(mu3)
print("Covariance:")
printMatrix(estimatedC3)
print("Eigenvalues:")
print(lambda1)
print(lambda2)
print("Eigenvectors:")
print(v1)
print(v2)
plotContours(Y3, mu3, estimatedC3, "Question 3")

print("QUESTION 4")
print("")
muClass1 = [0, 0]
muClass2 = [3, 3]
CClass1 = [[2, 0.5], [0.5, 1]]
CClass2 = [[1, -0.3], [-0.3, 2]]
Y1 = generateClass(muClass1, CClass1)
Y2 = generateClass(muClass2, CClass2)
inverse1 = invert_matrix(CClass1)
inverse2 = invert_matrix(CClass2)
det1 = (CClass1[0][0] * CClass1[1][1] - CClass1[0][1] * CClass1[1][0])
det2 = (CClass2[0][0] * CClass2[1][1] - CClass2[0][1] * CClass2[1][0])
plt.figure(figsize=(9, 9))
plt.scatter(Y1[0], Y1[1], s=10, alpha=0.35, label="Class 1")
plt.scatter(Y2[0], Y2[1], s=10, alpha=0.35, label="Class 2")
minX = min(min(Y1[0]), min(Y2[0])) - 1
maxX = max(max(Y1[0]), max(Y2[0])) + 1
minY = min(min(Y1[1]), min(Y2[1])) - 1
maxY = max(max(Y1[1]), max(Y2[1])) + 1
numberOfPoints = 200
xValues = []
yValues = []
for i in range(numberOfPoints):
    xValues.append(minX + (maxX - minX) * i / (numberOfPoints - 1))
    yValues.append(minY + (maxY - minY) * i / (numberOfPoints - 1))
boundary = []
for y in yValues:
    row = []
    for x in xValues:
        g1 = discriminant(x, y, muClass1, inverse1, det1)
        g2 = discriminant(x, y, muClass2,inverse2, det2)
        row.append(g1 - g2)
    boundary.append(row)
plt.contour(xValues, yValues, boundary, levels=[0])
plt.scatter(muClass1[0], muClass1[1], marker="x", s=150, linewidths=3)
plt.scatter(muClass2[0], muClass2[1], marker="x", s=150, linewidths=3)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Question 4 Decision Boundary")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()