import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import sys
from sklearn.preprocessing import StandardScaler
import pickle
source_loc = "/project/wyin/jlee/ml-project/source"
sys.path.append(source_loc)
from utilities import LitNeuralNet, LitDataModule, ProblemStatement

step_num = int(sys.argv[1]) - 1
num_cpus = int(sys.argv[2])
prob_file = sys.argv[3]
save_loc = sys.argv[4]
log_name = sys.argv[5]

torch.set_num_threads(num_cpus // 2)    #Pytorch doesn't use hyperthreadindg, so use half the cpus
data_loc, X_name, y_name = ProblemStatement(problem_file = prob_file).prob_vars


#Set hyperparameters using pickle file
grid = pickle.load(open(os.path.join(save_loc, "grid-search"), "rb"))

hparams = ['layer_sizes', 'learning_rate', 'batch_size', 'schedule_factor']
sigs = [1] * len(hparams)
digs = [0] * len(hparams)

for i in range(len(hparams) - 1, 0, -1):
    sigs[i - 1] = sigs[i] * len(grid[hparams[i]]) 

for i in range(len(hparams)):
    digs[i] = step_num // sigs[i]
    step_num %= sigs[i]

hparam_dict = dict([(hparam, grid[hparam][digs[i]]) for i, hparam in enumerate(hparams)])
print(hparam_dict)


######################################################
#Create log and validation folders if they don't yet exist

log_folder = os.path.join(save_loc, "logs")
val_folder = os.path.join(save_loc, "val-ends")
log_path = os.path.join(log_folder, log_name)
val_path = os.path.join(val_folder, log_name)

if step_num == 0:
    
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)
       
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    
    print("made paths")
    
model_name = f"{hparam_dict['layer_sizes']}, {hparam_dict['learning_rate']}, {hparam_dict['batch_size']}, {hparam_dict['schedule_factor']}"
    
data_module = LitDataModule(data_loc, hparam_dict['batch_size'], X_name = X_name, y_name = y_name)

logger = TensorBoardLogger(log_path, name = model_name)

trainer = pl.Trainer(enable_checkpointing=grid['save_models'], max_time=grid['max_time'], logger = logger, enable_progress_bar = False)
model = LitNeuralNet(hparam_dict['layer_sizes'], lr = hparam_dict['learning_rate'], lr_factor = hparam_dict['schedule_factor'])
trainer.fit(model, datamodule=data_module)


end_res = trainer.validate(model, dataloaders = data_module)
f = open(f"{val_path}/{model_name}","wb")

pickle.dump(end_res,f)
f.close()