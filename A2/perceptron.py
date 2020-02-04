# import numpy as np
import torch.nn as nn
import torch


# FIXME: It's showing a warning about differences between target size and input size
class Perceptron(nn.Module):
	"""
	Simplest unit of Neural Networks
	"""

	def __init__(self, input_size, output_size, learning_rate):
		super(Perceptron, self).__init__()
		
		self.learning_rate = learning_rate
		self.layer = nn.Linear(input_size, output_size)

	def forward(self, data, check_data):
		"""
		Feed data to Perceptron. Calculate the estimated output and Cost function
		using Sigmoid as activation function 
		"""
		# Estimate the output
		activation_fn = nn.Sigmoid()
		self.output = activation_fn(self.layer(data))
		# Cost function calculation
		self.loss = nn.BCELoss()
		self.cost = self.loss(self.output, check_data)

	def back_prop(self):
		"""
		Correction of weights and bias 
		"""
		# Declare the optimizer
		optimizer = torch.optim.SGD(Perceptron.parameters(self), self.learning_rate)
		
		# Reset the gradients
		optimizer.zero_grad()
		
		# Calculate the gradients
		self.cost.backward()
		
		# Update parameters
		optimizer.step()

	# TODO: Rewrite without using numpy
	def print_metrics(self, check_data, data_samples):
		"""
		Metrics calculation and printing
		"""
		# check_data_TF = check_data_TF = np.where(check_data > 0, True, False)

		# TP = np.where(self.output >= 0.5, self.output, 0) * check_data_TF
		# TP = len(TP[TP > 0])

		# FP = np.where(self.output >= 0.5, self.output, 0) * ~check_data_TF
		# FP = len(FP[FP > 0])

		# TN = np.where(self.output < 0.5, self.output, 0) * ~check_data_TF
		# TN = len(TN[TN > 0])

		# FN = np.where(self.output < 0.5, self.output, 0) * check_data_TF
		# FN = len(FN[FN > 0])

		# print('TPs: {:^3} FPs: {:^3} FNs: {:^3} TNs: {:^3}'.format(TP, FP, FN, TN), end = ' | ')

		# accuracy = (TP + TN) / data_samples
		# precision = TP / (TP + TN)
		# recall = TP / (TP + FN)
		# F1 = (2 * precision * recall) / (precision + recall)

		# print('Exactitud: {:.2E}. Precision: {:.2E}. Recall: {:.2E}. F1: {:.2E}'.format(accuracy, precision, recall, F1))
