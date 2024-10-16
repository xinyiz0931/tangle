"""
Configuration class: config_type = ["train", 'infer']
Author: xinyi
Date: 20220517
Use it to load default config file by `cfg = Config(config_type="train")`
"""
import os
import yaml
from pathlib import Path
# import platform

class Config(object):
    def __init__(self, config_type, config_file=None, config_data=None):
        
        self.config_type = config_type
        self.root_dir = Path(__file__).parent.parent

        if config_data is None and config_file is None:
            # get a fixed path to `config.yaml`
            # config_path = os.path.realpath(
            #     os.path.join(self.root_dir, "cfg/config.yaml"))
            config_path = os.path.join(self.root_dir, "cfg", "config.yaml")
            data = self.load(config_path)
        
        elif config_file is not None and config_data is None:
            # get a path to config_file
            config_path = os.path.join(self.root_dir, "cfg", config_file)
            data = self.load(config_path)

        else:
            data = config_data

        print(data)
        # ignore platform
        # if platform.system() == "Linux":
        #     root_dir = data["root_dir_linux"]
        # elif platform.system() == "Windows":
        #     root_dir = data["root_dir_win"]

        try:
            self.config = data[self.config_type]
        except KeyError:
            raise AttributeError

        if self.config_type == "train":
            self.save_dir = os.path.join(self.root_dir, "checkpoints", self.save_folder)
            self.data_dir = os.path.join(self.root_dir, "data", self.dataset)

        elif self.config_type == "infer":
            self.picknet_ckpt = os.path.join(self.root_dir, "checkpoints", *self.pick_ckpt_folder)
            self.pullnet_ckpt = os.path.join(self.root_dir, "checkpoints", *self.pull_ckpt_folder)

        else:
            print(f"Wrong config type: {self.config_type} ... ")

    def load(self, config_path):
        try:
            with open(config_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)

        except FileNotFoundError:
            print("Wrong file or file path")
        return data

    def __getattr__(self, key):
        try:
            return self.config[key]
        except KeyError:
            raise AttributeError

    def display(self):
        print("-"*75)
        for a in self.config.keys():
            print("{:30} {}".format(a, self.config[a]))
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))
        print("-"*75)

    def record(self, log_path):
        with open(log_path, "w") as f:
            for a in self.config.keys():
                print("{:30} {}".format(a, self.config[a]), file=f)
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                    print("{:30} {}".format(a, getattr(self, a)), file=f)
            print("\n")
