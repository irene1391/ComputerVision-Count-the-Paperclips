import os
import torch


def get_config():
    return Config()

class Config(object):
    def __init__(self):
        # training configuration
        self.epochs = 200
        self.batch_size = 128
        self.num_workers = 8
        self.lr = 1e-3
        self.lr_decay = 0.95

        # model configuration
        self.save_frequency = 5
        self.val_frequency = 2

        self.set_network_structure()

        self.valid_rate = 0.8
        self.data_path = "../clips/"
        self.label_path = "../"
        self.model_dir = "weight/"
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def set_network_structure(self):
        self.cfg = [32, 64, 'M', 128, 256, 'M']
        self.fc = [512, 512, 76]