#Need this for matrix operations
import numpy as np


def gradient_descent(x, y, n, m, alpha, max_iter):
    theta = np.ones(n) #Initialize theta as all 1
    for i in range(0, max_iter): #Using max number of iterations
        print("Iteration ", i, " Current theta ", theta)
        h = np.dot(x, theta) #First, calculate the hypothesis
        #Then the gradient, this is the hardest step to understand
        #What happens here is we take the formula, it's the sum as follows
        #sum = 0
        #for i in range(0, m): sum += (h0(x[i]) - y[i]) * x[i]
        #with x[0] = 1, then sum/m
        #But, we have numpy! and numpy.dot, so we just use that
        #and also we use all the h0s already calculated before
        gradient = np.dot((h - y), x) / m
        theta -= alpha * gradient #Lastly we update theta
    return theta

def read_table(filename, n, m):
    x = np.zeros(shape = (m, n))
    y = np.zeros(shape = m)
    f = open(filename, 'r')
    i = 0
    for line in f:
        aux = line.split()
        x[i][0] = 1
        for j in range(1, n):
            x[i][j] = aux[j]
        y[i] = aux[n]
        i += 1
    print(x)
    print(y)
    return x, y

def normalize(x, y, n):
    for i in range(1, n):
        x[:, i] = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
    y = (y - np.mean(y)) / np.std(y)
    return x, y

n = 2
m = 62
x, y = read_table('data1.txt', n, m)
x, y = normalize(x, y, n)
theta = gradient_descent(x, y, n, m, 0.01, 20000)
print(theta)

n = 4
m = 20
x, y = read_table('data2.txt', n, m)
x, y = normalize(x, y, n)
theta = gradient_descent(x, y, n, m, 0.01, 20000)
print(theta)
