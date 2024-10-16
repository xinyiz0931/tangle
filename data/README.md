# Dataset

Download two datasets (`picknet_dataset.zip` and `pullnet_dataset.zip`) from [here](https://drive.google.com/drive/folders/1i_tRZcTMqNASh4RaOxcy0d9z5iqd5eup?usp=sharing), unzip and put them under `tangle/data/`, make sure the directory looks like this: 

```
tangle
├── cfg
├── data
│   ├── picknet_data (16GB)
│   └── pullnet_data (1.4GB)
├── ...
└── tools
```

Here is the detailed structures inside each dataset: 
```
        picknet_data
        ├── images
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── ...
        ├── masks
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── ...
        └── labels.npy - shape=(N,), 0: picking, 1: separating
```

```
        pullnet_data
        ├── _images (before rotation)
        │   ├── 000000.png
        │   └── ...
        ├── _masks (before rotation)
        │   ├── 000000.png
        │   └── ...
        ├── _positions.npy (before rotation) - shape=(N,2)
        ├── _directions.npy (before rotation) - shape=(N,2)
        ├── images (after rotation, in use)
        │   ├── 000000.png
        │   └── ...
        ├── masks (after rotation, in use)
        │   ├── 000000.png
        │   └── ...
        └── positions.npy (after rotation, in use) - shape=(N,2)
```