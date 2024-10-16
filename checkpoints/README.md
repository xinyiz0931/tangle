
# Checkpoints

This is the output directory for saving the trained checkpoints. You can also download our test checkpoints for inference. 

Download two files `picknet_test_ckpt.pth` and `pullnet_test_ckpt.pth` from [here](https://drive.google.com/drive/folders/1i_tRZcTMqNASh4RaOxcy0d9z5iqd5eup?usp=sharing), put them under `tangle/checkpoints/` and make sure the sturcture looks like this: 


```
tangle
├── cfg
├── checkpoints
│   ├── picknet_ckpt (16GB)
|   │   └── picknet_test_ckpt.ckpt
│   └── pullnet_ckpt (1.4GB)
|   │   └── pullnet_test_ckpt.ckpt 
├── ...
└── tools