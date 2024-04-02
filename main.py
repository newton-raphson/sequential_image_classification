from configs.config_reader import Configuration
from executor.executor import Executor
import sys
if __name__ == '__main__':
    # pass the config file path to the function
    config_file_path = sys.argv[1]
    print("Running with config file: ",config_file_path)
    config = Configuration(config_file_path)
    executor = Executor(config) 
    executor.run()