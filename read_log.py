import ast
import matplotlib.pyplot as plt
import numpy as np

from utils.train_config import configs


starting_epoch = 0

with open(configs.log_filename, 'r') as f:
    res = ast.literal_eval(f.read())

num_epochs = len(res[0])
plt.plot(res[3][starting_epoch:num_epochs], label = 'Validation Set ' + configs.model_name, linewidth=1.5, color='m', linestyle = 'dotted')
plt.plot(res[2][starting_epoch:num_epochs], label = 'Training Set ' + configs.model_name, linewidth=1.5, color='m')



plt.ylabel('EER', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.grid()
plt.ylim([0,1])
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(fontsize=10)
plt.title('')
plt.show()


t = 0
