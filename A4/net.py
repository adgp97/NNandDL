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
# 		- fourth column represents the negative slope when leaky ReLU is selected

	def __init__(self, layers, learning_rate):
		"""
		Constructor
		"""
		super(Net, self).__init__()

		self.learning_rate = learning_rate
		# self.mode = mode
		self.layers = nn.ModuleList()
		self.act_funcs = nn.ModuleList()

		for i in range(layers.shape[0]):

			self.layers.append(nn.Linear(int(layers[i][0]), int(layers[i][1])))

			if layers[i][2] == None:
				self.act_funcs.append(None)	# No act func
			elif layers[i][2] == 'sigmo':
				self.act_funcs.append(nn.Sigmoid())
			elif layers[i][2] == 'tanh':
				self.act_funcs.append(nn.Tanh())
			elif layers[i][2] == 'relu':
				self.act_funcs.append(nn.ReLU())
			elif (layers[i][2] == 'leakyrelu') and (len(layers[i]) == 4):
				self.act_funcs.append(nn.LeakyReLU(layers[i][3]))
			else:
				print("Activation function not defined. Aborting...")
				sys.exit()


	def forward(self, data, check_data):
		"""
		Feed data to Network. Calculate the estimated output and Cost function
		"""
		self.output = data	# DO NOT ERRASE. Do not pay attentition to this
		for i in range(len(self.layers)):

			try:
				self.output = self.act_funcs[i](self.layers[i](self.output))

			except TypeError:
				# This should happen when activation function is set to None
				self.output = self.layers[i](self.output)

		if torch.is_tensor(check_data):
			loss_fn = nn.BCELoss()
			self.cost = loss_fn(self.output, check_data.view(len(check_data),1))


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

	def print_metrics(self, check_data):
		"""
		Metrics calculation and printing
		"""
		ones_tensor = torch.ones(len(check_data), 1)
		zeros_tensor = torch.zeros(len(check_data), 1)
		check_data_TF = torch.where(check_data > 0, ones_tensor, zeros_tensor)[1,:].view(len(check_data),1)
		check_data_TF_neg = torch.where(check_data > 0, zeros_tensor, ones_tensor)[1,:].view(len(check_data),1)

		TP = torch.where(self.output >= 0.5, self.output, zeros_tensor) * check_data_TF
		TP = len(TP[TP > 0])

		FP = torch.where(self.output >= 0.5, self.output, zeros_tensor) * check_data_TF_neg
		FP = len(FP[FP > 0])

		TN = torch.where(self.output < 0.5, self.output, zeros_tensor) * check_data_TF_neg
		TN = len(TN[TN > 0])

		FN = torch.where(self.output < 0.5, self.output, zeros_tensor) * check_data_TF
		FN = len(FN[FN > 0])

		print('M: {} | TPs: {:^3} FPs: {:^3} FNs: {:^3} TNs: {:^3}'.format(check_data.shape[0], TP, FP, FN, TN), end = ' | ')

		accuracy = (TP + TN) / check_data_TF.shape[0]
		precision = TP / (TP + TN)
		recall = TP / (TP + FN)
		F1 = (2 * precision * recall) / (precision + recall)

		print('Exactitud: {:.2E}. Precision: {:.2E}. Recall: {:.2E}. F1: {:.2E}'.format(accuracy, precision, recall, F1))