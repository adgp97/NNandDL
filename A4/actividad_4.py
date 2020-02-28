import matplotlib
# matplotlib.use('agg')   # Comment for plot
import matplotlib.pyplot as plt
import torch
import numpy as np
import utils
from net import Net
import os.path

file_name = 'sonar.csv'

epoch_num = 5000
learning_rate = 0.00005
layers_drop = np.asarray([[60,  80,  'relu', 0, 0.5], 
					      [80, 100,  'relu', 0, 0.5], 
					      [100, 80,  'relu', 0, 0.5], 
					      [80,  60,  'relu', 0, 0.5], 
					      [60,  40,  'relu', 0, 0.5], 
					      [40,   1, 'sigmo', 0 ,  0]])
layers_no_drop = np.asarray([[60,  80,  'relu', 0, 0], 
					      	 [80, 100,  'relu', 0, 0], 
					      	 [100, 80,  'relu', 0, 0], 
					      	 [80,  60,  'relu', 0, 0], 
					      	 [60,  40,  'relu', 0, 0], 
					      	 [40,   1, 'sigmo', 0, 0]])

# Cost function initialization
cost_acc_drop = 0
cost_acc_no_drop = 0
cost_train_drop = []
cost_val_drop = []
cost_train_no_drop = []
cost_val_no_drop = []

# Metrics arrays for plotting purposes
recall_drop = []
recall_no_drop = []
acc_drop = []
acc_no_drop = []

# Best metrics (metric values when the model is saved)
recall_drop_saved = 0
recall_no_drop_saved = 0
acc_drop_saved = 0.8		# Initialized high to obtain better results
acc_no_drop_saved = 0.8		# IDEM
cost_val_min_drop = 100
cost_val_min_no_drop = 100 

# Epoch the model was saved on
mark_drop_saved = 0
mark_no_drop_saved = 0

TP_drop = []
TN_drop = []
FP_drop = []
FN_drop = []

TP_no_drop = []
TN_no_drop = []
FP_no_drop = []
FN_no_drop = []

batch_sizes = 31

# Only shuffle the data once
if not os.path.exists('shuffled_' + file_name):
    utils.shuffle_dataset(file_name)

train, val, test = utils.load_shuffled_dataset('shuffled_' + file_name, batch_sizes)

model_drop = Net(layers_drop, learning_rate)
model_no_drop = Net(layers_no_drop, learning_rate)

# Training
for curr_ep in range(epoch_num):
	
	# Switching to training mode (dropout enabling)
	model_drop.train()
	model_no_drop.train()

	# Resetting cost accumulators
	cost_acc_drop = 0
	cost_acc_no_drop = 0

	# Training using minibatch
	for __, batch in enumerate(train):

		model_drop.forward(batch[:,:60], batch[:,60])
		model_no_drop.forward(batch[:,:60], batch[:,60])

		model_drop.back_prop('adam', weight_decay = 0)
		model_no_drop.back_prop('adam', weight_decay = 0)

		cost_acc_drop = cost_acc_drop + model_drop.cost
		cost_acc_no_drop = cost_acc_no_drop + model_no_drop.cost

	# Collect current errors
	cost_train_drop.append(cost_acc_drop / len(train))
	cost_train_no_drop.append(cost_acc_no_drop / len(train))

	# Switching to evaluation mode (dropout disabling)
	model_drop.eval()
	model_no_drop.eval()

	# Calculate the cost of the validation set
	model_drop.forward(val[:,:60], val[:,60].view(len(val),1))
	model_no_drop.forward(val[:,:60], val[:,60].view(len(val),1))
	
	# Collect current errors with the validation set
	cost_val_drop.append(model_drop.cost)
	cost_val_no_drop.append(model_no_drop.cost)

	# Collect current metrics for plotting purpouses
	model_drop.calc_metrics(val[:,60])
	recall_drop.append(model_drop.recall)
	acc_drop.append(model_drop.accuracy)
	TP_drop.append(model_drop.TP)
	TN_drop.append(model_drop.TN)
	FP_drop.append(model_drop.FP)
	FN_drop.append(model_drop.FN)
	
	# Save model if improved
	if model_drop.recall >= recall_drop_saved and model_drop.accuracy >= acc_drop_saved and model_drop.cost <= cost_val_min_drop:
		torch.save(model_drop.state_dict(), 'model_drop')
		recall_drop_saved = model_drop.recall
		acc_drop_saved = model_drop.accuracy
		cost_val_min_drop = model_drop.cost
		mark_drop_saved = curr_ep	# Mark the epoch the model was saved on


	# Collect current metrics for plotting purpouses
	model_no_drop.calc_metrics(val[:,60])
	recall_no_drop.append(model_no_drop.recall)
	acc_no_drop.append(model_no_drop.accuracy)
	TP_no_drop.append(model_no_drop.TP)
	TN_no_drop.append(model_no_drop.TN)
	FP_no_drop.append(model_no_drop.FP)
	FN_no_drop.append(model_no_drop.FN)

	# Save model if improved
	if model_no_drop.recall >= recall_no_drop_saved and model_no_drop.accuracy >= acc_no_drop_saved and model_no_drop.cost <= cost_val_min_no_drop:
		torch.save(model_no_drop.state_dict(), 'model_no_drop')
		recall_no_drop_saved = model_no_drop.recall
		acc_no_drop_saved = model_no_drop.accuracy
		cost_val_min_no_drop = model_no_drop.cost
		mark_no_drop_saved = curr_ep	# Mark the epoch the model was saved on

	print('Epoch: ', curr_ep)
	print('Case dropout. Cost train: {:.4E} | Cost validation: {:.4E} | Accuracy: {:.4E} | Recall: {:.4E}'.format(cost_train_drop[curr_ep], cost_val_drop[curr_ep],model_drop.accuracy, model_drop.recall))
	# model_drop.print_metrics()
	print('Case no dropout. Cost train: {:.4E} | Cost validation: {:.4E} | Accuracy: {:.4E} | Recall: {:.4E}'.format(cost_train_no_drop[curr_ep], cost_val_no_drop[curr_ep], model_no_drop.accuracy, model_no_drop.recall))
	# model_no_drop.print_metrics()


# PLOT 1
y_drop = np.zeros(epoch_num)
y_drop[mark_drop_saved] = max(max(cost_val_drop), max(cost_train_drop))

plt.subplot(1,2,1)	# Plot costs with dropout
plt.plot(range(epoch_num), cost_train_drop,    label = 'Training')
plt.plot(range(epoch_num), cost_val_drop,      label = 'Validation')
plt.stem(range(epoch_num), y_drop,             label = 'Checkpoint',    linefmt = 'm--', markerfmt = 'xm')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Case with dropout')
plt.legend()
plt.grid()


y_no_drop = np.zeros(epoch_num)
y_no_drop[mark_no_drop_saved] = max(max(cost_train_no_drop), max(cost_val_no_drop))

plt.subplot(1,2,2)	# Plot costs without dropout
plt.plot(range(epoch_num), cost_train_no_drop, label = 'Training')
plt.plot(range(epoch_num), cost_val_no_drop,   label = 'Validation')
plt.stem(range(epoch_num), y_no_drop,          label = 'Checkpoint', linefmt = 'k--', markerfmt = 'ok')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Case without dropout')
plt.legend()
plt.grid()

plt.suptitle('Cost functions vs Epoch number')
# plt.savefig('cost')
plt.show()




# PLOT 2
y_drop = np.zeros(epoch_num)
y_drop[mark_drop_saved] = max(max(recall_drop), max(acc_drop))

plt.subplot(1,2,1)	# Metrics with dropout
plt.plot(range(epoch_num), recall_drop,        label = 'Recall')
plt.plot(range(epoch_num), acc_drop,           label = 'Accuracy')
plt.stem(range(epoch_num), y_drop,             label = 'Checkpoint',    linefmt = 'm--', markerfmt = 'xm')
plt.xlabel('Epochs')
plt.ylabel('Metric values')
plt.title('Case with dropout')
plt.legend()
plt.grid()

y_no_drop = np.zeros(epoch_num)
y_no_drop[mark_no_drop_saved] = max(max(recall_no_drop), max(acc_no_drop))

plt.subplot(1,2,2)	# Metrics without dropout
plt.plot(range(epoch_num), recall_no_drop,     label = 'Recall')
plt.plot(range(epoch_num), acc_no_drop,        label = 'Accuracy')
plt.stem(range(epoch_num), y_no_drop,          label = 'Checkpoint', linefmt = 'k--', markerfmt = 'ok')
plt.xlabel('Epochs')
plt.ylabel('Metric values')
plt.title('Case without dropout')
plt.legend()
plt.grid()

plt.suptitle('Metrics vs Epoch number')
# plt.savefig('metrics')
plt.show()




# PLOT 3
y_drop = np.zeros(epoch_num)
y_drop[mark_drop_saved] = max(max(TP_drop), max(TN_drop), max(FP_drop), max(FN_drop))

plt.subplot(1,2,1)	# Possitives y negatives with dropout
plt.plot(range(epoch_num), TP_drop,   label = 'TP')
plt.plot(range(epoch_num), TN_drop,   label = 'TN')
plt.plot(range(epoch_num), FP_drop,   label = 'FP')
plt.plot(range(epoch_num), FN_drop,   label = 'FN')
plt.stem(range(epoch_num), y_drop,    label = 'Checkpoint', linefmt = 'm--', markerfmt = 'xm')
plt.xlabel('Epochs')
plt.ylabel('Confussion matrix values')
plt.title('Case with dropout')
plt.legend()
plt.grid()


y_no_drop = np.zeros(epoch_num)
y_no_drop[mark_no_drop_saved] = max(max(TP_no_drop), max(TN_no_drop), max(FP_no_drop), max(FN_no_drop))

plt.subplot(1,2,2)	# Possitives y negatives without dropout
plt.plot(range(epoch_num), TP_no_drop,   label = 'TP')
plt.plot(range(epoch_num), TN_no_drop,   label = 'TN')
plt.plot(range(epoch_num), FP_no_drop,   label = 'FP')
plt.plot(range(epoch_num), FN_no_drop,   label = 'FN')
plt.stem(range(epoch_num), y_no_drop,    label = 'Checkpoint', linefmt = 'k--', markerfmt = 'ok')
plt.xlabel('Epochs')
plt.ylabel('Confussion matrix values')
plt.title('Case without dropout')
plt.legend()
plt.grid()

plt.suptitle('Confussion matrix values vs Epoch number')
# plt.savefig('confmat')
plt.show()

# RESULTS
print('Best metrics')
print('Case dropout. Recall: {} | Accuracy: {} | Epoch: {}'.format(recall_drop_saved, acc_drop_saved, mark_drop_saved))
print('Case no dropout. Recall: {} | Accuracy: {} | Epoch: {}'.format(recall_no_drop_saved, acc_no_drop_saved, mark_no_drop_saved))


# LOAD BEST MODELS
model_drop.load_state_dict(torch.load('model_drop'))
model_drop.eval()
model_no_drop.load_state_dict(torch.load('model_no_drop'))
model_no_drop.eval()

# Reloading datasets bc now train must be a single batch
train, val, test = utils.load_shuffled_dataset('shuffled_' + file_name, batch_size = 124)

for __, batch in enumerate(train):
	print('TRAINING SET')
	print("Confussion matrix of model with dropout")
	model_drop.forward(batch[:,:60], batch[:,60])
	model_drop.print_metrics()
	print("Confussion matrix of model without dropout")
	model_no_drop.forward(batch[:,:60], batch[:,60])
	model_no_drop.print_metrics()

print('VALIDATION SET')
print("Confussion matrix of model with dropout")
model_drop.forward(val[:,:60], val[:,60])
model_drop.print_metrics()
print("Confussion matrix of model without dropout")
model_no_drop.forward(val[:,:60], val[:,60])
model_no_drop.print_metrics()


print('TEST SET')
print("Confussion matrix of model with dropout")
model_drop.forward(test[:,:60], test[:,60])
model_drop.print_metrics()
print("Confussion matrix of model without dropout")
model_no_drop.forward(test[:,:60], test[:,60])
model_no_drop.print_metrics()
