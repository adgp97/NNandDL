import torch.nn as nn
import torch
import sys

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
			loss_fn = nn.MSELoss()
			self.cost = loss_fn(self.output, check_data)


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