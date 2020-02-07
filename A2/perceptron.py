import torch.nn as nn
import torch
import sys

class Perceptron(nn.Module):
	"""
	Simplest unit of Neural Networks
	"""

	def __init__(self, input_size, output_size, learning_rate, mode):
		"""
		Constructor
		"""
		super(Perceptron, self).__init__()
		
		self.learning_rate = learning_rate
		self.mode = mode
		self.layer = nn.Linear(input_size, output_size)

	def forward(self, data, check_data):
		"""
		Feed data to Perceptron. Calculate the estimated output and Cost function
		using Sigmoid as activation function 
		"""
		if self.mode == 1:

			# Estimate the output
			activation_fn = nn.Sigmoid()
			self.output = activation_fn(self.layer(data))
			# Cost function calculation
			loss_fn = nn.BCELoss()
			self.cost = loss_fn(self.output, check_data)

		elif self.mode == 2:

			# Estimate the output
			self.output = self.layer(data)
			# Cost function calculation
			loss_fn = nn.MSELoss()
			self.cost = loss_fn(self.output, check_data)

		else:
			print('Error: Mode not supported. Aborting...')
			sys.exit()

	def back_prop(self):
		"""
		Correction of weights and bias 
		"""
		# Declare the optimizer
		optimizer = torch.optim.SGD(self.parameters(), self.learning_rate)
		
		# Reset the gradients
		optimizer.zero_grad()
		
		# Calculate the gradients
		self.cost.backward()
		
		# Update parameters
		optimizer.step()