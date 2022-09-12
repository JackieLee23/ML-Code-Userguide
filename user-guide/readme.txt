This userguide provides templates for running Jackson Lee's ML code, and 

Before you start running code, your data should be saved in a folder, with three separate npz files: train-set.npz, val-set.npz, and test-set.npz. Each file should have input parameters stored as an n by m numpy array, and corresponding ground truth outputs stored as an n by k numpy array. 

In the problem-definition.txt file: change the data folder location, name of input features, and name of outputs, to match your data. The problem-definition.txt file must be included in the same folder as the code templates for the code to run. 

This userguide has four templates:
    - Baseline errors: Calculates the baseline errors in the test set
    - KNN: Does both hyperparameter tuning and testing set evaluation for the K-nearest-neighbors algorithm
    - Neural net hyperparameter tuning: Provides template for tuning hyperparameters of a neural network
    - Neural net evaluation: Provides template for evaluating neural net on test set
    
The problem demonstrated in this userguide is: given input DOS, predict the corresponding t', t'' , and J. This userguide can be applied to other datasets by simply copying the userguide folder, replacing the names in the problem-definition.txt file, and retraining the ML models.  