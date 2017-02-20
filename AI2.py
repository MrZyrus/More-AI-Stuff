#Need this for matrix operations
import numpy as np


def gradient_descent(x, y, n, m, alpha, max_iter,filename):
    theta = np.ones(n) #Initialize theta as all 1
    f = open(filename,'w')
    for i in range(0, max_iter): #Using max number of iterations
        h = np.dot(x, theta) #First, calculate the hypothesis
        error = np.sum((h - y)**2) / 2*m 
        s = str(i) + " " + str(error) + "\n"
        f.write(s)
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
theta = gradient_descent(x, y, n, m, 0.1, 100,"res_D1_0.1.txt")
print(theta)

n = 4
m = 20
x, y = read_table('data2.txt', n, m)
x, y = normalize(x, y, n)
theta = gradient_descent(x, y, n, m, 0.1, 50,"res_D2_0.1.txt")
print(theta)
theta = gradient_descent(x, y, n, m, 0.3, 50,"res_D2_0.3.txt")
print(theta)
theta = gradient_descent(x, y, n, m, 0.5, 50,"res_D2_0.5.txt")
print(theta)
theta = gradient_descent(x, y, n, m, 0.7, 50,"res_D2_0.7.txt")
print(theta)
theta = gradient_descent(x, y, n, m, 0.9, 50,"res_D2_0.9.txt")
print(theta)
theta = gradient_descent(x, y, n, m, 1.0, 50,"res_D2_1.0.txt")
print(theta)
