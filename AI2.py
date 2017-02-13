#Need this for matrix operations
import numpy as np
import random
import math

def gradient_descent(x, y, n, m, alpha, max_iter):
    theta = np.ones(n) #Initialize theta as all 1
    for i in range(0, max_iter): #Using max number of iterations
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
    return x, y

def normalize(v,n):
    m = 0;# Media
    for i in v:
        m += i
    m = m/n
    sd = 0# Desviacion standard
    for j in v:
        sd+= math.pow(j-m,2)
    sd = sd/n
    sd = math.sqrt(sd)
    l=0
    for k in v: # Calculo los valores normalizados
        k = (k - m)/sd
        v[l] = k
        l+=1
    return v

n = 2
m = 62
x, y = read_table('test.txt', n, m)
y = normalize(y,m);
theta = gradient_descent(x, y, n, m, 0.1, 10)
print(theta)
