from configs.config_reader import Configuration
from executor.executor import Executor
from evaluations import evaluate
from dataloader.data_loader import load_test_data
from model.networks import AutoEncoder
import sys
if __name__ == '__main__':
    # pass the config file path to the function
    config_file_path = sys.argv[1]
    print("Running with config file: ",config_file_path)
    config = Configuration(config_file_path)
    executor = Executor(config) 
    if config.post_process:
        test_dataloader = load_test_data(data_path=config.data_path,config=config,device=executor.device)
        if config.train_encoder:
            model = executor.model
            model = Executor.load_model_ae(model,executor.model_save_path)
            evaluate.evaluate_autoencoder(model,executor.postprocess_save_path,executor.device,test_dataloader)
        else:
            model = executor.model
            model = Executor.load_model_ae(model,executor.model_save_path)
            auto_encoder = AutoEncoder()
            auto_encoder = Executor.load_model_ae(auto_encoder, config.ae_path, best=True)  

            evaluate.evaluate_sequential(model,auto_encoder,executor.postprocess_save_path,executor.device,test_dataloader)
    else:
        executor.run()