import numpy as np

class Perceptron:
	"""
	Simplest unit of Neural Networks
	"""

	def __init__(self, input_size, learning_rate):
		self.input_size = input_size
		self.learning_rate = learning_rate

		self.weights = np.random.uniform(low = -1, high = 1, size = self.input_size)
		self.bias = np.random.uniform(low = -1, high = 1, size = 1)

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

	def print_metrics(self, check_data, data_samples):
		"""
		Metrics calculation and printing
		"""
		check_data_TF = check_data_TF = np.where(check_data > 0, True, False)

		TP = np.where(self.output >= 0.5, self.output, 0) * check_data_TF
		TP = len(TP[TP > 0])

		FP = np.where(self.output >= 0.5, self.output, 0) * ~check_data_TF
		FP = len(FP[FP > 0])

		TN = np.where(self.output < 0.5, self.output, 0) * ~check_data_TF
		TN = len(TN[TN > 0])

		FN = np.where(self.output < 0.5, self.output, 0) * check_data_TF
		FN = len(FN[FN > 0])

		print('TPs: {:^3} FPs: {:^3} FNs: {:^3} TNs: {:^3}'.format(TP, FP, FN, TN), end = ' | ')

		accuracy = (TP + TN) / data_samples
		precision = TP / (TP + TN)
		recall = TP / (TP + FN)
		F1 = (2 * precision * recall) / (precision + recall)

		print('Exactitud: {:.2E}. Precision: {:.2E}. Recall: {:.2E}. F1: {:.2E}'.format(accuracy, precision, recall, F1))
