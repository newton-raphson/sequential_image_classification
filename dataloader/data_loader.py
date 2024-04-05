import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils.files import create_directory
import numpy as np
import h5py
import glob
import numpy as np


def load_data(data_path, config, device):
    """
    Load data from CSV files and prepare it for training.

    Args:
        data_path (str): The path to the data directory.
        config (Config): The configuration object.
        device (torch.device): The device to load the data onto.

    Returns:
        training_dataloader (torch.utils.data.DataLoader): The data loader for training.
        validation_dataset (torch.utils.data.TensorDataset): The dataset for validation.
    """
    save_directory=data_path
    # get the file with *.mat from the data path
    # data_file = glob.glob("*.mat")
    # load the data as follows

    data = h5py.File(data_path,'r')

    # get train_x
    train_x = np.array(data.get("train_set_x"))
    train_x = train_x
    # get train_y
    train_y = np.array(data.get("train_set_y"))
    train_y = train_y
    print(f"The length of train_X is {len(train_x)}")
    print(f"The length of train_Y is {len(train_y)}")
    # we need to transpose train_x 
    train_x = train_x.T
    train_x= train_x.reshape(-1,1,250,100)
    # train_x = np.array([image / np.max(image) for image in train_x])


    print(f"The shape is {train_x.shape}")

        # get train_x
    valid_x = np.array(data.get("valid_set_x"))
    # get train_y
    valid_y = np.array(data.get("valid_set_y"))
    
    valid_y = valid_y
    print(f"The length of valid_X is {len(train_x)}")
    print(f"The length of valid_Y is {len(train_y)}")

    # we need to transpose train_x 
    valid_x = valid_x.T
    valid_x= valid_x.reshape(-1,1,250,100)
    # valid_x = np.array([image / np.max(image) for image in valid_x])
    
    print("We are here bro")
    X = torch.tensor(train_x, dtype=torch.float32)
    Y = torch.tensor(train_y, dtype=torch.float32)
    val_X = torch.tensor(valid_x, dtype=torch.float32)
    val_Y = torch.tensor(valid_y, dtype=torch.float32)
    print("We are here bro")

    training_dataset = torch.utils.data.TensorDataset(X, Y)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=config.batchsize, shuffle=True)
    validation_dataset = torch.utils.data.TensorDataset(val_X, val_Y)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batchsize, shuffle=True)
    
    print("We are here bro")
    return training_dataloader, validation_dataloader

def load_test_data(data_path,config,device):
    """
    Load test data from a file and prepare it for evaluation.

    Args:
        data_path (str): The path to the data file.

    Returns:
        test_dataloader (torch.utils.data.DataLoader): The data loader for test data.
    """
    # Load data from the file
    data = h5py.File(data_path, 'r')

    # Get test_x
    test_x = np.array(data.get("test_set_x"))
    test_x = test_x # Normalize if needed

    # Get test_y
    test_y = np.array(data.get("test_set_y"))
    test_y = test_y

    # Transpose and reshape test_x
    test_x = test_x.T
    test_x = test_x.reshape(-1, 1, 250, 100)

    # test_x = np.array([image / np.max(image) for image in test_x])


    # Convert to PyTorch tensors
    X = torch.tensor(test_x, dtype=torch.float32)
    Y = torch.tensor(test_y, dtype=torch.float32)

    # Create a TensorDataset and DataLoader
    test_dataset = torch.utils.data.TensorDataset(X, Y)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batchsize, shuffle=False)

    return test_dataloader

