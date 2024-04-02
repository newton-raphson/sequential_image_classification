# This file reads config.ini and returns a Configuration object
# The Configuration object contains all the parameters needed for the training

import configparser


class Configuration:
    def __init__(self, file_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(file_path)

        # FILE
        self.directory = self.config.get("Files","directory")
        self.data_path = self.config.get("Files","data_path")
        self.expname = self.config.get("Files","experiment")
  
        


        # TRAIN PARAMS
        self.lr = self.config.getfloat("Training", "lr")
        self.epochs = self.config.getint("Training","epochs")
        self.minepochs = self.config.getint("Training","min_epochs")
        self.batchsize = self.config.getint("Training","batch_size")
        self.checkpointing = self.config.getint("Training","checkpointing")
        self.contd = self.config.getboolean("Training","continue")
        self.patience = self.config.getint("Training","patience")
    
    