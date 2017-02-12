#Need this for matrix operations
import numpy as np

def gradient_descent(x, y, n, m, theta, alpha, max_iter):
    theta = np.ones(n) #Initialize theta as all 1
    for i in range(0, max_iter): #Using max number of iterations
        h = np.dot(theta, x) #First, calculate the hypothesis
        
        #Then the gradient, this is the hardest step to understand
        #What happens here is we take the formula, it's the sum as follows

        #sum = 0
        #for i in range(0, m): sum += (h0(x[i]) - y[i]) * x[i]
        #with x[0] = 1, then sum/m

        #But, we have numpy! and numpy.dot, so we just use that
        #and also we use all the h0s already calculated before

        gradient = np.dot((h0 - y), x) / m 
        theta -= alpha * gradient #Lastly we update theta
    return theta