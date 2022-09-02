"""
Configuration class: config_type = ["train", 'infer']
Author: xinyi
Date: 20220517
Use it to load default config file by `cfg = Config(config_type="train")`
Otherwise, create a dictionary and use `cfg = Config(config_type="train", config_data=config_data)`
    config_data = {
        "root_dir_win": "C:\\Users\\xinyi\\Documents",
        "root_dir_linux": "/home/hlab/Documents/",
        "infer": 
        {
            "net_type": "pick_sep",
            "use_cuda": True,
            "batch_size": 1,
            "img_height": 512,
            "img_width": 512,
            "pick_ckpt_folder": ["try8","model_epoch_10.pth"],
            "sepp_ckpt_folder": ["try_38","model_epoch_4.pth"],
            "sepd_ckpt_folder": ["try_new_res","model_epoch_12.pth"]
        }
    }
"""
import os
import yaml
import platform

class Config(object):
    def __init__(self, config_type, config_data=None):
        
        self.config_type = config_type

        if config_data is None:
            # get a fixed path to `config.yaml`
            dir_path = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.realpath(
                os.path.join(dir_path, "../cfg/config.yaml"))
            data = self.load(config_path)

        else:
            data = config_data
        if platform.system() == "Linux":
            root_dir = data["root_dir_linux"]
        elif platform.system() == "Windows":
            root_dir = data["root_dir_win"]
        self.root_dir = root_dir

        try:
            self.config = data[self.config_type]
        except KeyError:
            raise AttributeError

        if self.config_type == "train":
            self.save_dir = os.path.join(root_dir, "Checkpoint", self.save_folder)
            self.data_dir = os.path.join(root_dir, "Dataset", self.dataset)

        elif self.config_type == "infer":
            self.pick_ckpt = os.path.join(root_dir, "Checkpoint", *self.pick_ckpt_folder)
            self.sepp_ckpt = os.path.join(root_dir, "Checkpoint", *self.sepp_ckpt_folder)
            if self.sep_type == "vector": 
                self.sepd_ckpt = os.path.join(root_dir, "Checkpoint", *self.sepd_ckpt_folder_v)
            elif self.sep_type == "spatial":
                self.sepd_ckpt = os.path.join(root_dir, "Checkpoint", *self.sepd_ckpt_folder_s)

            if "pick" in self.net_type:
                self.dataset_dir = os.path.join(root_dir, "Dataset", "picknet", 'test')
            elif "sep" in self.net_type:
                self.dataset_dir = os.path.join(root_dir, "Dataset", "sepnet", 'test')
            else:
                print(f"Wrong inference type: {self.net_type} ... ")
        elif self.config_type == "validate":
            self.pick_ckpt = os.path.join(root_dir, "Checkpoint", self.pick_ckpt_folder)
            self.sephold_ckpt = os.path.join(root_dir, "Checkpoint", self.sepp_ckpt_folder)
            self.sepdir_ckpt = os.path.join(root_dir, "Checkpoint", self.sepd_ckpt_folder_v)
            self.seppull_ckpt = os.path.join(root_dir, "Checkpoint", self.sepd_ckpt_folder_s)

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
