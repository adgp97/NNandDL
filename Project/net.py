import torch.nn as nn
import torch
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

		self.optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)#, weight_decay=weight_decay)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer = self.optimizer, step_size = 10, gamma = 0.1)

	def forward(self, data, check_data, weights):
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
		

		loss_fn = nn.CrossEntropyLoss(weight = weights)
		self.cost = loss_fn(self.output, check_data)


	def back_prop(self):
		"""
		Correction of weights and bias
		"""

		# Reset the gradients
		self.optimizer.zero_grad()

		# Calculate the gradients
		self.cost.backward()

		# Update parameters
		self.optimizer.step()

	def calc_metrics(self, check_data):
		"""
		Metrics calculation and printing
		"""

		self.prediction = torch.argmax(self.output, dim = 1)

		self.accuracy = accuracy_score(check_data, self.prediction)

		self.precision, self.recall, self.F1, __ = precision_recall_fscore_support(check_data, self.prediction, average = 'micro')
		
	def print_metrics(self):

		# print('TPs: {:^3} FPs: {:^3} FNs: {:^3} TNs: {:^3}\n'.format(self.TP, self.FP, self.FN, self.TN))
		print('Exactitud: {:.2E}. Precision: {:.2E}. Recall: {:.2E}. F1: {:.2E}'.format(self.accuracy, self.precision, self.recall, self.F1))
