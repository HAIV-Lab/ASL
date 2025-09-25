# Revisiting Out-of-Distribution Detection: Angular Separation Learning as a Powerful and Simple Baseline

This project is the code implementation of the paper *Revisiting Out-of-Distribution Detection: Angular Separation Learning as a Powerful and Simple Baseline*, and the entire code is built on top of the official implementation of OpenOOD.

# Environment
The versions of packages in our environment are detailed in the `requirements.txt` file.

# Datasets
For dataset preparation, please refer to the dataset preparation section of OpenOOD and organize your datasets according to the following structure:
```
├── ...
├── data
│   ├── benchmark_imglist
│   ├── images_classic
│   └── images_largescale
├── openood
├── results
│   ├── checkpoints
│   └── ...
├── scripts
├── main.py
├── ...
```

# Training and Evaluation
We have placed the training and evaluation scripts in the `scripts/ood/asl` folder; for details, please refer to the files within this folder.