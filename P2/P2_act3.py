from random import random
from math import exp

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs,sum_error):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
	return sum_error

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

def calcular_prueba(network,prueba):
        fn  = 0
        fp = 0
        sum_error = 0
        for row in prueba:
                outputs = forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                if row[-1] == 0 and outputs.index(max(outputs)) == 1:
                        fp+=1
                if row[-1] == 1 and outputs.index(max(outputs)) == 0:
                        fn+=1
        return sum_error,fn,fp


def read_data(filename):
    data = []
    f = open(filename, 'r')
    for line in f:
        row = []
        aux = line.split(',')
        
        if aux[-1].strip() == "Iris-setosa":
            aux[-1] = 1
            aux.append(int('1'))
        else:
            aux[-1] = 0
            aux.append(int('0'))
        for i in aux:
            row.append(float(i))
        row[-1] = int(row[-1])
        data.append(row)
    return data

def calcular_prueba(network,prueba):
        fn  = 0
        fp = 0
        sum_error = 0
        for row in prueba:
                outputs = forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                if row[-1] == 0 and outputs.index(max(outputs)) == 1:
                        fp+=1
                if row[-1] == 1 and outputs.index(max(outputs)) == 0:
                        fn+=1
        return sum_error,fn,fp

    
n_inputs = 5
n_outputs = 2
network = []
index = 0
neurons = 4
data = read_data("iris-50-train.txt")
prueba = read_data("iris-50-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 50 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp = calcular_prueba(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos negativos: %d' % fn)
        print('Falsos positivos: %d' %fp)
        index+=1
        neurons+=1
neurons = 4
data = read_data("iris-60-train.txt")
prueba = read_data("iris-60-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 60 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp = calcular_prueba(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos negativos: %d' % fn)
        print('Falsos positivos: %d' %fp)
        index+=1
        neurons+=1
neurons = 4
data = read_data("iris-70-train.txt")
prueba = read_data("iris-70-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 70 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp = calcular_prueba(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos negativos: %d' % fn)
        print('Falsos positivos: %d' %fp)
        index+=1
        neurons+=1
neurons = 4
data = read_data("iris-80-train.txt")
prueba = read_data("iris-80-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 80 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp = calcular_prueba(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos negativos: %d' % fn)
        print('Falsos positivos: %d' %fp)
        index+=1
        neurons+=1
neurons = 4
data = read_data("iris-90-train.txt")
prueba = read_data("iris-90-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 90 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp = calcular_prueba(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos negativos: %d' % fn)
        print('Falsos positivos: %d' %fp)
        index+=1
        neurons+=1



def read_data2(filename):
    data = []
    f = open(filename, 'r')
    for line in f:
        row = []
        aux = line.split(',')
        
        if aux[-1].strip() == "Iris-setosa":
            aux[-1] = 1
            aux.append(int('1'))
        elif aux[-1].strip() == "Iris-versicolor":
            aux[-1] = 0
            aux.append(int('0'))
        else:
            aux[-1] = 2
            aux.append(int('2'))
        for i in aux:
            row.append(float(i))
        row[-1] = int(row[-1])
        data.append(row)
    return data
    
def calcular_prueba2(network,prueba):
        fn  = 0
        fp = 0
        fs = 0
        sum_error = 0
        for row in prueba:
                outputs = forward_propagate(network, row)
                expected = [0 for i in range(n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                if row[-1] != 0 and outputs.index(max(outputs)) == 0:
                        fp+=1
                if row[-1] != 1 and outputs.index(max(outputs)) == 1:
                        fn+=1
                if row[-1] != 2 and outputs.index(max(outputs)) == 2:
                        fs+=1
        return sum_error,fn,fp,fs

n_outputs = 3
network = []
index = 0
neurons = 4
data = read_data2("iris-50-train.txt")
prueba = read_data2("iris-50-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 50 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp,fs = calcular_prueba2(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos setosa: %d' % fn)
        print('Falsos versicolor: %d' % fp)
        print('Falsos virginica: %d' % fs)
        index+=1
        neurons+=1
neurons = 4
data = read_data2("iris-60-train.txt")
prueba = read_data2("iris-60-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 60 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp,fs = calcular_prueba2(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos setosa: %d' % fn)
        print('Falsos versicolor: %d' % fp)
        print('Falsos virginica: %d' % fs)
        index+=1
        neurons+=1
neurons = 4
data = read_data2("iris-70-train.txt")
prueba = read_data2("iris-70-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 70 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp,fs = calcular_prueba2(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos setosa: %d' % fn)
        print('Falsos versicolor: %d' % fp)
        print('Falsos virginica: %d' % fs)
        index+=1
        neurons+=1
neurons = 4
data = read_data2("iris-80-train.txt")
prueba = read_data2("iris-80-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 80 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp,fs = calcular_prueba2(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos setosa: %d' % fn)
        print('Falsos versicolor: %d' % fp)
        print('Falsos virginica: %d' % fs)
        index+=1
        neurons+=1
neurons = 4
data = read_data2("iris-90-train.txt")
prueba = read_data2("iris-90-test.txt")
for i in range(7):
        network.append(initialize_network(n_inputs, neurons, n_outputs))
        print('Configuacion de dataset con 90 y %d neuronas: ' % neurons)
        print('Error de entrenamiento: %.3f' % train_network(network[index], data, 0.1, 1000, n_outputs,0))
        perror,fn,fp,fs = calcular_prueba2(network[index], prueba)
        print('Error de prueba: %.3f' % perror)
        print('Flasos setosa: %d' % fn)
        print('Falsos versicolor: %d' % fp)
        print('Falsos virginica: %d' % fs)
        index+=1
        neurons+=1
