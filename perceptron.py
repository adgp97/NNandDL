# __init__: constructor.
# Parametros: numero de entradas, learning rate(?), numero de epocas(?)
# Funciones: inicializacion aleatoria de los pesos y del bias 

# feedforward: evaluar 
# Parametros: pesos, bias, entrada
# Funciones: multiplicar entradas por pesos y sumar, sumar bias, evaluar la funcion de activacion

# backpropagation: ajustar parametros
# Parametros: pesos, bias, learning rate
# Funciones: calcular el gradiente, actualizar los pesos y el bias

import numpy as np

class Perceptron:
	"""
	Simplest unit of Neural Networks
	"""

	def __init__(self, input_size, learning_rate):
		self.input_size = input_size
		self.learning_rate = learning_rate

		self.weights = np.random.uniform(low = -0.5, high = 0.5, size = self.input_size)
		self.bias = np.random.uniform(low = -1, high = 1, size = 1)

# Data es un arreglo cuyo numero de columnas es el numero de entradas
# del perceptron y el numero de filas es el numero de muestras
	def feed(self, data, check_data):
		"""
		Feed data to Perceptron. Calculate the estimated output and Cost function
		using Sigmoid as activation function 
		"""
		# Estimate the output
		self.output = 1/( 1 + np.exp(-1 * (np.dot(data,self.weights) + self.bias)))
		# Cost function calculation
		self.cost = -1*np.average(check_data * np.log(self.output) + (1 - check_data)*np.log(1 - self.output))
	
	def grad_desc(self, data, check_data):
		"""
		Gradient Descent calculation
		"""
		self.dw = np.dot(data.T,self.output - check_data) / data.shape[0]
		self.db = np.average(self.output - check_data)

	def back_prop(self):
		"""
		Correction of weights and bias 
		"""
		self.weights -= self.learning_rate * self.dw
		self.bias -= self.learning_rate * self.db
