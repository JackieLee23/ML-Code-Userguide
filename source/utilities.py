import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
import os
from os import path
from sklearn.neighbors import KNeighborsRegressor
import pickle


class LitNeuralNet(pl.LightningModule):
    def __init__(self, layer_sizes, lr = 0.01, lr_factor = 0.0):
        super(LitNeuralNet, self).__init__()
        
        modules = []
        for i in range(len(layer_sizes) - 1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            if i != len(layer_sizes) - 2:
                modules.append(nn.ReLU())
        
        self.forward_prop = nn.Sequential(*modules)
        self.learning_rate = lr
        self.factor = lr_factor
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.forward_prop(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        
        # Forward pass
        predicted = self(X)
        loss = F.mse_loss(predicted, y)
        
        #log to tensorboard
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        
        # Forward pass
        predicted = self(X)
        loss = F.mse_loss(predicted, y)
        
        #log to tensorboard
        self.log("val_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.factor == 0.0:
            return optimizer
        
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.factor, min_lr = 1e-7)
            return {
                "optimizer":optimizer,
                "lr_scheduler" : {
                    "scheduler" : sch,
                    "monitor" : "train_loss",

                }
            }

        
class LitDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.size(dim=0)
    
    
class LitDataModule(pl.LightningDataModule):
    def __init__(self, data_loc, batch_size, X_name, y_name):
        super().__init__()
        self.data_loc = data_loc
        self.batch_size = batch_size
        self.X_name = X_name
        self.y_name = y_name
    
    def setup(self, stage: str = None):
        data = ScaledData(self.data_loc, self.X_name, self.y_name)
        
        if stage == "fit" or stage is None:
            self.train_dataset = LitDataset(data.train_X, data.train_y)
            self.val_dataset = LitDataset(data.val_X, data.val_y)
            
        if stage == "test" or stage is None:
            self.test_dataset = LitDataset(data.test_X, data.test_y)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=2)

#Gets all the data from a data file for you, assuming you have train-set, val-set, and test-set npz files
class UnscaledData:
    def __init__(self, data_loc, X_name, y_name, check_data = False):
        train_set = np.load(path.join(data_loc, 'train-set.npz'))
        self.train_X = train_set[X_name]
        self.train_y = train_set[y_name]
        
        val_set = np.load(path.join(data_loc, 'val-set.npz'))
        self.val_X = val_set[X_name]
        self.val_y = val_set[y_name]
        
        test_set = np.load(path.join(data_loc, 'test-set.npz'))
        self.test_X = test_set[X_name]
        self.test_y = test_set[y_name]
        
        if check_data:
            print("Double checking dataset sizes:\n")
            print(f"Training set x size: {self.train_X.shape}. Training set y size: {self.train_y.shape}\n")
            print(f"validation set x size: {self.val_X.shape}. Validation set y size: {self.val_y.shape}\n")
            print(f"Testing set x size: {self.test_X.shape}. Testing set y size: {self.test_y.shape}\n")

#Same as above, but scales data for you
class ScaledData:
    def __init__(self, data_loc, X_name, y_name, check_data = False):
        unscaled = UnscaledData(data_loc, X_name, y_name)
        self.train_y = unscaled.train_y
        self.val_y = unscaled.val_y
        self.test_y = unscaled.test_y

        scaler = StandardScaler()
        self.train_X = scaler.fit_transform(unscaled.train_X)
        self.val_X = scaler.transform(unscaled.val_X)
        self.test_X = scaler.transform(unscaled.test_X)
        
        if check_data:
            print("Double checking dataset sizes:\n")
            print(f"Training set x size: {self.train_X.shape}. Training set y size: {self.train_y.shape}\n")
            print(f"validation set x size: {self.val_X.shape}. Validation set y size: {self.val_y.shape}\n")
            print(f"Testing set x size: {self.test_X.shape}. Testing set y size: {self.test_y.shape}\n")
  

#Loads problem statement parameters
class ProblemStatement:
    def __init__(self, problem_file = None):
        if problem_file == None:
            problem_file = "problem-definition.txt"
        
        self.prob_vars = []
        with open(problem_file) as f:
            for line in f:
                line = line.partition('#')[0]
                line = line.rstrip()
                self.prob_vars.append(line)
                
        f.close()
        
#see_results helper       
def plot_one(ax, mse_params, index, x_vals, tick_labels):
    
    ax.plot(x_vals, mse_params[index][2], label = "Ground Truth")
    ax.plot(x_vals, mse_params[index][1], label = "ML Predicted")
    
    if tick_labels != None:
        ax.set_xticks(x_vals, ('t1', 't2', 'J'))
    ax.legend()
   
#Get mse betweeen predicted and ground truth, and see graphs of worst performing percentiles. Returns
#Predicted and ground truth values for those percentiles
def see_results(predicted, truth, grid_shape, percentiles, x_vals, tick_labels = None, same_ylim = True):
    mse_mat = (predicted - truth) ** 2
    mse_list = np.mean(mse_mat, axis = 1)
    print(f"model mse: {np.mean(mse_list)}")

    mse_params = zip(mse_list, predicted, truth)
    mse_params = sorted(mse_params, key = lambda x: x[0], reverse = True)
    mse_percentiles = []
    
    dim1, dim2 = grid_shape
    fig, ax = plt.subplots(dim1, dim2, figsize = (15, dim1 * 5))
    
    for i in range(dim1):
        for j in range(dim2):
            if i * dim2 + j < len(percentiles):
                percentile = percentiles[i * dim2 + j]
                index = percentile * (len(mse_params)//100)
                if same_ylim:
                    ax[i][j].set_ylim(np.min(truth), np.max(truth))
                    
                plot_one(ax[i][j], mse_params, index, x_vals, tick_labels)
                ax[i][j].set_title(f"{percentile} percentile")
                
                mse_percentiles.append(mse_params[index])
                
    return mse_percentiles
                
#Return mean error and std
def baseline_errors(test_y):
    means = np.mean(test_y, axis = 0)

    full_means = np.tile(means, (test_y.shape[0],1))
    
    errors = (test_y - full_means) ** 2
    errors_list = np.mean(errors, axis = 1) #mse for every spectrum
        
    return np.mean(errors_list), np.std(errors_list)

#Return best knn
def find_knn(data_loc, X_name, y_name, max_neighbors = 20, weight_func = 'uniform', scaled = True):
    
    if scaled:
        data = ScaledData(data_loc, X_name, y_name)
        
    else:
        data = UnscaledData(data_loc, X_name, y_name)
        
    results = []
    for i in range(1, max_neighbors):
        print(f"trying n neights = {i}")
        neigh = KNeighborsRegressor(n_neighbors = i, weights = weight_func)
        neigh.fit(data.train_X, data.train_y)
        predicted = neigh.predict(data.val_X)

        mse = np.mean((data.val_y - predicted)**2)
        results.append(mse)
        
    plt.plot(np.arange(1, max_neighbors), results)
    best_n = np.argmin(results) + 1
    print(f"Minimum val loss: {np.min(results)}")
    print(f"Best number of neighbors: {best_n}")
    
    best_model = KNeighborsRegressor(n_neighbors = best_n, weights = weight_func)
    best_model.fit(data.train_X, data.train_y)
    return best_model


#Returns ordered (validation loss, model string) pairs
def order_validation(val_end_path):
    errors = []
    for file in os.listdir(val_end_path):
        if file != ".ipynb_checkpoints":
            file_loc = os.path.join(val_end_path, file)
            x = pickle.load(open(file_loc, "rb"))
            errors.append((x[0]['val_loss'], file))
            
    errors = sorted(errors, key = lambda x : x[0])
    return errors

#Get checkpointed model from a tensorboard log
def get_model(model_loc):
    check_dir = os.path.join(model_loc, "version_0/checkpoints")
    check_file = os.path.join(check_dir, os.listdir(check_dir)[0])
    model = LitNeuralNet.load_from_checkpoint(check_file)
    return model

def get_prediction(model, inputs):
    input_tensor = torch.from_numpy(inputs).float()
    predicted = model(input_tensor).detach().numpy()
    return predicted