#Need this for matrix operations
import numpy as np
import math

def gradient_descent(x, y, n, m, alpha, max_iter):
    theta = np.ones(n) #Initialize theta as all 1
    for i in range(0, max_iter): #Using max number of iterations
        #print("Iteration ", i, " Current theta ", theta)
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
        #Solo se utilizo para el analisis de datos los rasgos de Año de construccion,
        #tamaño en square feet, cantidad de baños y cantidad de habitaciones.
        x[i][0] = 1
        x[i][1] = aux[20]
        x[i][2] = aux[47]
        x[i][3] = aux[50]
        x[i][4] = aux[52]
        y[i] = aux[81]
        i += 1
    return x, y

def clean_data(filename):
    m = 0
    f = open(filename, 'r')
    f2 = open('cleaned.txt', 'w')
    for line in f:
        aux = line.split()
        #Se eliminaron los datos que no contenian los 82 rasgos,
        #Los que tenian la caracteristica (all) y (arg) en MS Zoning
        #Los que contenian la caracteristica 'Normal' en Sale Condition
        #Y los que tenian un tamano en Square Feet mayor a 1500
        if len(aux) == 82 and '(all)' not in aux and '(agr)' not in aux and 'Normal' in aux and int(aux[47]) <= 1500:
            for item in aux:
                f2.write("%s " % item)
            f2.write("\n")
            
            m+=1
    return 'cleaned.txt',m
            
def normalize(x, y, n):
    for i in range(1, n):
        x[:, i] = (x[:, i] - np.mean(x[:, i])) / np.std(x[:, i])
    y = (y - np.mean(y)) / np.std(y)
    return x, y

def bias(theta,x,y):
    yhat = np.dot(x,theta)
    avg = sum(yhat-y)/len(yhat)
    return avg

def max_dev(theta,x,y):
    yhat = np.dot(x,theta)
    return max(abs(y-yhat))

def mean_abs(theta,x,y):
    yhat = np.dot(x,theta)
    return sum(abs(y-yhat))/len(yhat)

def mean_sq(theta,x,y):
    yhat = np.dot(x,theta)
    return sum(pow(y-yhat,2))/len(yhat)

n = 5
#Se realiza una limpieza de datos
cleaned_file,m  = clean_data('data_act3.txt')
#Se analizan los nuevos datos para normalizar y calcular el modelo 
x, y = read_table(cleaned_file, n, m)
x, y = normalize(x, y, n)
theta = gradient_descent(x, y, n, m, 0.01, 20000)
#Se escogen los grupos de prueba y entrenamiento
ex = x[0:670]
px = x[671:839]
ey = y[0:670]
py = y[671:839]
print('Prueba Bias, grupo entrenamiento:')
print(bias(theta,ex,ey))
print('Prueba Bias, grupo prueba:')
print(bias(theta,px,py))
print('Prueba Maximun Deviation, grupo entrenamiento:')
print(max_dev(theta,ex,ey))
print('Prueba Maximun Deviation, grupo prueba:')
print(max_dev(theta,px,py))
print('Prueba Mean Absolute Deviation, grupo entrenamiento:')
print(mean_abs(theta,ex,ey))
print('Prueba Mean Absolute Deviation, grupo prueba:')
print(mean_abs(theta,px,py))
print('Prueba Mean Square Error, grupo entrenamiento:')
print(mean_sq(theta,ex,ey))
print('Prueba Mean Square Error, grupo prueba:')
print(mean_sq(theta,px,py))
