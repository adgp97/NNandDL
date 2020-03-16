import torch.nn as nn
import torch
import sys
import numpy as np

class Net(nn.Module):
	"""
	Neural Networks class
	"""
# Layers:
# 		- each row represents a layer
# 		- first column represents the number of inputs
# 		- second column represents the number of outputs
# 		- third column represents the activation function. None, sigmo, tanh, relu, leakyrelu
# 		- fourth column represents the negative slope when leaky ReLU is selected. If not selected, this value is ignored
# 		- fifth column represents layer dropout probability

	def __init__(self, layers, learning_rate):
		"""
		Constructor
		"""
		super(Net, self).__init__()

		self.learning_rate = learning_rate
		self.layers = nn.ModuleList()
		self.act_funcs = nn.ModuleList()
		self.drop = nn.ModuleList()

		for i in range(layers.shape[0]):

			self.layers.append(nn.Linear(int(layers[i][0]), int(layers[i][1])))
			self.drop.append(nn.Dropout(float(layers[i][4])))

			if layers[i][2] == None:
				self.act_funcs.append(None)	# No act func
			elif layers[i][2] == 'sigmo':
				self.act_funcs.append(nn.Sigmoid())
			elif layers[i][2] == 'tanh':
				self.act_funcs.append(nn.Tanh())
			elif layers[i][2] == 'relu':
				self.act_funcs.append(nn.ReLU())
			elif layers[i][2] == 'leakyrelu':
				self.act_funcs.append(nn.LeakyReLU(layers[i][3]))
			elif layers[i][2] == 'softmax':
				self.act_funcs.append(nn.Softmax(dim=1))
			else:
				print("Activation function not defined. Aborting...")
				sys.exit()


	def forward(self, data, check_data):
		"""
		Feed data to Network. Calculate the estimated output and Cost function
		"""
		self.output = data	# DO NOT ERRASE. Do not pay attentition to this

		for i in range(len(self.layers)):
			# if (i < len(self.layers) - 1) and

			if (i == len(self.layers) - 1) and self.training:
			 	# When the last layer is reacehd and if it is in traning mode, the activation function is not applied
				self.output = self.drop[i](self.layers[i](self.output))

			else:
				try:
					self.output = self.drop[i](self.act_funcs[i](self.layers[i](self.output)))

				except TypeError:
					# This should happen when activation function is set to None
					self.output = self.drop[i](self.layers[i](self.output))

		loss_fn = nn.CrossEntropyLoss()
		self.cost = loss_fn(data, check_data)

		

	def back_prop(self, opt, momentum = 0, weight_decay=0):
		"""
		Correction of weights and bias
		"""

		# Declare the optimizer
		if   opt == 'sgd':
			optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate, momentum=momentum, weight_decay=weight_decay)
		elif opt == 'rmsprop':
			optimizer = torch.optim.RMSprop(self.parameters(), self.learning_rate, weight_decay=weight_decay)
		elif opt == 'adam':
			optimizer = torch.optim.Adam(self.parameters(), self.learning_rate, weight_decay=weight_decay)
		else:
			print('Optimizer not defined. Aborting...')
			sys.exit()

		# Reset the gradients
		optimizer.zero_grad()

		# Calculate the gradients
		self.cost.backward()

		# Update parameters
		optimizer.step()

	def calc_metrics(self, check_data):
		"""
		Metrics calculation and printing
		"""
		# TODO: update using scikit learn
		ones_tensor = torch.ones(len(check_data), 1)
		zeros_tensor = torch.zeros(len(check_data), 1)
		check_data_TF = torch.where(check_data > 0, ones_tensor, zeros_tensor)[1,:].view(len(check_data),1)
		check_data_TF_neg = torch.where(check_data > 0, zeros_tensor, ones_tensor)[1,:].view(len(check_data),1)

		self.TP = torch.where(self.output >= 0.5, self.output, zeros_tensor) * check_data_TF
		self.TP = len(self.TP[self.TP > 0])

		self.FP = torch.where(self.output >= 0.5, self.output, zeros_tensor) * check_data_TF_neg
		self.FP = len(self.FP[self.FP > 0])

		self.TN = torch.where(self.output < 0.5, self.output, zeros_tensor) * check_data_TF_neg
		self.TN = len(self.TN[self.TN > 0])

		self.FN = torch.where(self.output < 0.5, self.output, zeros_tensor) * check_data_TF
		self.FN = len(self.FN[self.FN > 0])

		# try:
		self.accuracy = (self.TP + self.TN) / check_data_TF.shape[0]
		self.precision = self.TP / (self.TP + self.TN)
		self.recall = self.TP / (self.TP + self.FN)
		# self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)

	def print_metrics(self):

		print('TPs: {:^3} FPs: {:^3} FNs: {:^3} TNs: {:^3}\n'.format(self.TP, self.FP, self.FN, self.TN))
		# print('Exactitud: {:.2E}. self.precision: {:.2E}. self.recall: {:.2E}. F1: {:.2E}'.format(self.accuracy, self.precision, self.recall, self.F1))
