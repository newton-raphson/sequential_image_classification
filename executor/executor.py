
import numpy as np
from utils.files import create_directory
import os
import sys
from configs.config_reader import Configuration
import torch
from dataloader.data_loader import load_data
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
import torch.nn as nn
from model.networks import Encoder,Classifier
from utils.pickling import CPU_Unpickler
import glob
from torch.optim.lr_scheduler import StepLR
from model.networks import Classifier,Encoder


class Executor:
    def __init__(self, config):
        self.config = config
        

        self.main_path = create_directory(os.path.join(config.directory,f"{self.config.expname}"))

        # this folder will contain all the models for particular number of data-points
        self.data_path = config.data_path

        self.model_path = create_directory(os.path.join(self.main_path,"classification"))
        #  this creates a folder inside the model path on the basis of
        #  the loss parameters supplied
        self.loss_path = create_directory(os.path.join(self.model_path,"loss_bce"))
        #  this creates a folder inside the loss path on the basis of
        #  the training parameters supplied
        self.train_path = create_directory(os.path.join(self.loss_path,f"lr_{config.lr},epochs_{config.epochs},min_epochs_{config.minepochs},batch_size_{config.batchsize}"))

        # inside the train_path create a folder to save the models for checkpointing
        self.model_save_path = create_directory(os.path.join(self.train_path,"models"))

        # inside the train_path create a folder to save the  post processing results
        self.postprocess_save_path = create_directory(os.path.join(self.train_path,"postprocess"))
        self.plot_save_path = create_directory(os.path.join(self.train_path,"plots"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define the model
        self.model = Classifier()
        self.loss = torch.nn.CrossEntropyLoss()

        print(f"Cuda is {torch.cuda.is_available()}")
        print("\nExecutor initialized\n")
    def train(self):


        training_dataloader, validation_dataloader = load_data(self.data_path,self.config,self.device)     
        self.model.to(self.device)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        # ##################### ADDED FOR TESTING ############################
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
        # ####################################################################
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        if self.config.contd:
            # load the best model from the model_save_path
            self.model, optimizer, start_epoch, loss_per_epoch, best_val_loss,val_loss_per_epoch =\
                Executor.load_model(self.model,optimizer,self.model_save_path,best=True)  
        else:
            start_epoch = 0
            loss_per_epoch = []
            val_loss_per_epoch = []
            best_val_loss = float('inf')

        self.model.train()
        counter = 0
        for i in range(start_epoch, int(self.config.epochs)):
            loss=0
            train_loss = 0
            val_loss = 0
            torch.cuda.empty_cache()
            for batch, (x_batch, y_batch) in enumerate(training_dataloader):
                loss = self.loss(x_batch.to(self.device), y_batch.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                del x_batch
                del y_batch
            # take a step in the scheduler
            scheduler.step()
            torch.cuda.empty_cache()
            train_loss = train_loss/len(training_dataloader)
            loss_per_epoch.append(train_loss)
            val_loss = 0
            self.model.eval()
            for batch, (x_batch, y_batch) in enumerate(validation_dataloader):
                loss = self.loss(x_batch.to(self.device), y_batch.to(self.device))
                val_loss += loss.item()
            val_loss = val_loss/len(validation_dataloader)
            val_loss_per_epoch.append(val_loss)
            self.model.train()
            # write this to a file 
            str_to_write = f"Epoch {i+1}/{self.config.epochs}: train loss {train_loss} validation loss {val_loss}\n"
            print(str_to_write)
            with open(os.path.join(self.train_path,"train_loss.txt"),"a") as f:
                f.write(str_to_write)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # save the model
                # code to save the model 
                Executor.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=True)
            else:
                # increase the counter by 1
                counter += 1
            if counter >= self.config.patience and i >= self.config.minepochs:
                print("Early stopping: No improvement for the last {} epochs".format(self.config.patience))
                Executor.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=False)
                break
            if i%self.config.checkpointing == 0:
                # save the model every 
                Executor.save_model(self.model, optimizer, loss_per_epoch, i,best_val_loss,val_loss_per_epoch,self.model_save_path,best=False)
                fig, ax = plt.subplots()
                ax.plot(loss_per_epoch, label='train_loss')
                ax.plot(val_loss_per_epoch, label='val_loss')
                ax.set_title('Loss vs Epochs')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend() 
                fig.savefig(os.path.join(self.plot_save_path, f"loss{i}.png"))
                plt.close(fig)

        # save the model every 
    @staticmethod
    def save_model(model, optimizer, loss_per_epoch, epoch,best_val_loss,val_loss_per_epoch,save_path,best=False):
        if best:
            checkpoint_data = {
                'epoch': epoch,
                'loss_per_epoch': loss_per_epoch,
                'best_val_loss': best_val_loss,
                'val_loss_per_epoch': val_loss_per_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_path = os.path.join(save_path, f"best_model.pkl")
        else:
            # just save the model state dict and epoch 
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }
            checkpoint_path = os.path.join(save_path, f"model_epoch{epoch}.pkl")
        with open(checkpoint_path, 'wb') as checkpoint_file:
            pickle.dump(checkpoint_data, checkpoint_file)   
    @staticmethod
    def load_model(model,optimizer,save_path,best = False):
        if best:
            checkpoint_path = os.path.join(save_path, "best_model.pkl")
            with open(checkpoint_path, 'rb') as checkpoint_file:
                if not torch.cuda.is_available():
                    print("No CUDA Function")
                    checkpoint_data = CPU_Unpickler(checkpoint_file).load()
                else:
                    checkpoint_data = pickle.load(checkpoint_file)
            model = Executor.model_device_handler(model,checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            epoch = checkpoint_data['epoch']+1
            best_val_loss = checkpoint_data['best_val_loss']
            val_loss_per_epoch = checkpoint_data['val_loss_per_epoch']
            loss_per_epoch = checkpoint_data['loss_per_epoch']
            return model, optimizer, epoch, loss_per_epoch, best_val_loss, val_loss_per_epoch
        else:
            print(f"Looking for model in {save_path}")
            model_files = glob.glob(save_path + '/model_epoch*.pkl')
            # print(f"Model files found {model_files}")
            if len(model_files) == 0:
                print("No model found")
                raise FileNotFoundError
            # pick the last model file
            # Sort the list of model files based on modification time
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            # Pick the last (most recent) model file
            last_model_file = model_files[0]

            checkpoint_path = last_model_file
            print(f"Loading model from {checkpoint_path}")
            print("Here")
            with open(checkpoint_path, 'rb') as checkpoint_file:
                if not torch.cuda.is_available():
                    print("No CUDA Function")
                    checkpoint_data = CPU_Unpickler(checkpoint_file).load()
                else:
                    checkpoint_data = pickle.load(checkpoint_file)
            model = Executor.model_device_handler(model,checkpoint_data['model_state_dict'])
            epoch = checkpoint_data['epoch']+1
            return model, epoch
    @staticmethod
    def model_device_handler(model,model_state_dict):
        """ Handle model for different device configurations

        Args:
            model: model to be handled

            model_state_dict: model state dict to be loaded
        """
        # if the model is single gpu model and state dict is multi-gpu
        # then remove the module from the keys
        # this will work if the model is not multi-gpu model
        # as well 
        if not torch.cuda.is_available():
            if next(model.parameters()).is_cuda:
                model = model.to('cpu')

            # Remove 'module.' prefix from keys if necessary
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model
        
        if not isinstance(model, torch.nn.DataParallel):
            new_state_dict = OrderedDict()
            # Create a new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model
        
        # if the model is multi-gpu model and state dict is single-gpu
        # then add the module from the keys
        elif isinstance(model, torch.nn.DataParallel):
            new_state_dict = OrderedDict()
            # Create a new OrderedDict that does not contain `module.`
            for k, v in model_state_dict.items():
                name = 'module.' + k if not k.startswith('module.') else k  # add `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model
   
        volume_size = (self.config.cubesize, self.config.cubesize, self.config.cubesize)  # Adjust this based on your requirements
        spacing = (2/volume_size[0], 2/volume_size[1], 2/volume_size[2])
        x = torch.linspace(-1, 1, volume_size[0], device=self.device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        # Reshape the coordinates to create a DataFrame
        coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)


        ###########################################################################################
        volume_size = (self.config.cubesize, self.config.cubesize, self.config.cubesize)  # Adjust this based on your requirements

        # Adjust the spacing based on the provided minimum and maximum values
        # spacing = (
        #     (1.818538e+03 - (-3.437500e+02)) / volume_size[0],
        #     (2.970000e+02 - (-9.598353e+02)) / volume_size[1],
        #     (9.867585e+02 - (-4.425856e+03)) / volume_size[2]
        # )

        # # Generate coordinates using torch.linspace and meshgrid
        # x = torch.linspace(-3.437500e+02, 1.818538e+03, volume_size[0])
        # y = torch.linspace(-9.598353e+02, 2.970000e+02, volume_size[1])
        # z = torch.linspace(-4.425856e+03, 9.867585e+02, volume_size[2])
        # Set new limits
        # new_limits = (-1, 1)

        # spacing = (
        #     (new_limits[1] - new_limits[0]) / volume_size[0],
        #     (new_limits[1] - new_limits[0]) / volume_size[1],
        #     (new_limits[1] - new_limits[0]) / volume_size[2]
        # )

        # # Generate coordinates using torch.linspace and meshgrid
        # x = torch.linspace(new_limits[0], new_limits[1], volume_size[0])
        # y = torch.linspace(new_limits[0], new_limits[1], volume_size[1])
        # z = torch.linspace(new_limits[0], new_limits[1], volume_size[2])


        # xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

        # # Reshape the coordinates to create a DataFrame
        # coordinates = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3).to(self.device)
        ###########################################################################################
        batch_size = self.config.ppbatchsize  # Adjust based on available memory
        sdf_values = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        print("Loading the best model\n\n")
        print(f"Loading model from {self.model_save_path}")
        print("\n\n")
        # self.model,_, epoch, _, _, _ = Executor.load_model(self.model,optimizer,self.model_save_path,True)
        self.model,epoch= Executor.load_model(self.model,optimizer,self.model_save_path,False)
        print(f"Model loaded from epoch {epoch}")
        # self.model.eval()
        # batch_size=1000
        with torch.no_grad():
            for i in range(0, coordinates.shape[0], batch_size):
                print(f"\nProcessing batch {i//batch_size}")
                batch_coordinates = coordinates[i:i + batch_size]
                batch_sdf = self.model(batch_coordinates.to(self.device)).to(torch.float32)
                # type(batch_sdf)
                sdf_values.append(batch_sdf)
                batch_sdf = batch_sdf.ravel()
                # how to reduce ram consumption here
                del batch_coordinates

        sdf_values = torch.cat(sdf_values)
        # Reshape the SDF values array to match the volume shape
        sdf_values = sdf_values.cpu().numpy().reshape(volume_size)
        verts, faces, normals, _ = marching_cubes(sdf_values, level=0.0,spacing=spacing)
        print(f"Mesh generated for cube size {self.config.cubesize}")
        if self.config.rescale:
            # save the mesh
            centroid = np.mean(verts)
            verts -= centroid
        # save the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        # mesh.export(os.path.join(save_directory, f"output_mesh{i}.stl"), file_type='stl')
        path_to_save = os.path.join(self.postprocess_save_path, f"{self.geometry_name}_resconstructed_{epoch}_cube_{self.config.cubesize}.stl")
        print(f"Saving mesh to {path_to_save}")
        mesh.export(path_to_save, file_type='stl') 
        plot_stl(path_to_save,os.path.join(self.postprocess_save_path, f"{self.geometry_name}_resconstructed_{epoch}_cube_{self.config.cubesize}.gif"))
    def run(self):
        print("Running the executor")
        self.train()
        return