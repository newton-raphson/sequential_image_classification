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
        # Train Type
        self.train_encoder = self.config.getboolean("Type","encoder")
        if(not self.train_encoder):
            self.ae_path = self.config.get("Type","ae_path")
            self.hidden_size = self.config.getint("Type","hidden_size")
            self.num_layers = self.config.getint("Type","num_layers")

        # TRAIN PARAMS
        self.lr = self.config.getfloat("Training", "lr")
        self.epochs = self.config.getint("Training","epochs")
        self.minepochs = self.config.getint("Training","min_epochs")
        self.batchsize = self.config.getint("Training","batch_size")
        self.checkpointing = self.config.getint("Training","checkpointing")
        self.contd = self.config.getboolean("Training","continue")
        self.patience = self.config.getint("Training","patience")

        # POST PROCESS
        self.post_process = self.config.getboolean("PostProcess","post_process")

    
    