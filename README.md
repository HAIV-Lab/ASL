<article align="center" style="margin-bottom: 20px;">
    <h1 
        align="center"
        itemprop="title"
        style="font-size: 30px; font-weight: bold; margin-bottom: 20px;"
    >
    Why Feature Magnitude Deceives OOD Detectors: An Angular Separation Perspective
    </h1>
</article>

<div
    align="center"
    style="font-size: 18px; margin-bottom: 20px;"
>
    <a href="https://vain222.github.io/" target='_blank'>Hanlin Li</a>&emsp;
    <a href="https://jimm0011.github.io/" target='_blank'>Jing Ma</a>&emsp;
    Zehang Wei&emsp; 
    Jiamin Yan&emsp;
    <a href="https://eglxiang.github.io/" target='_blank'>Xiang Xiang</a>
</div>

<div 
    align="center"
    style="font-size: 16px; margin-bottom: 20px;"
>
Huazhong University of Science and Technology (HUST)&emsp;


Correspondence to Xiang Xiang (xex@hust.edu.cn)

</div>



This project is the code implementation of the paper *Why Feature Magnitude Deceives OOD Detectors: An Angular Separation Perspective*, and the entire code is built on top of the official implementation of OpenOOD.

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

# Acknowledgment
We would like to thank the OpenOOD team for their open-source implementation of OpenOOD.
