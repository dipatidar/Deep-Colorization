# Deep-Colorization

## Overview
This project aimed at designing and training a CNN regressor for automatic colorization of grayscale face images. The project is written in pytorch. 

## Project Requirement
The project requirement will be found at [here](https://github.com/dipatidar/Deep-Colorization/blob/main/PartII_DeepColorization.pdf).

## Project Report
The detailed report will be found at [here](https://github.com/dipatidar/Deep-Colorization/blob/main/PROJECT%202%20REPORT.pdf).

## Installation and Environment creation
Conda - 4.6.13
pytorch - 1.8.1+cu111 <br/>
numpy - 1.19.5 <br/>
pandas - 0.25.1 <br/>
scipy - 1.5.4 <br/>
python -  3.7.4 <br/>

Use the following command to create a conda environment from our 'req.txt' file:
```shell
conda create --name colorize_net_custom --file req.txt
```

Use the following command to create custom Juptyer Notebook kernel with conda environment:
```shell
 python -m ipykernel install --user --name=colorize_net_custom
```

## Dependencies
[python 3.7.7](https://www.python.org/downloads/release/python-374/)

[pytorch 1.8.1+cu11](https://pytorch.org/get-started/previous-versions/)

## Dataset
The dataset is present [here](https://github.com/dipatidar/Deep-Colorization/tree/main/DataSet).

## Program Execution

Unzip our project in a CUDA enabled environment. Run the following files in order:
```tex
1. DataAugmentationWithVal.ipynb 
2. Regressor_training_testing.ipynb 
3. ColorizeEarlyStop.ipynb
```

## Outputs

1. #### Regressor

   The regressor outputs two scalar values which are the average of the `a` and `b` channel for each of the images.

2. #### Colorizer

   The colorizer output color images and stores the same in the respective `output` folder based on the activation function. The grayscale images are stored under the **gray** folder and the colorized images goes into the **color** folder.

   

## GPU

The code extracts the `device` at the start of execution using ` torch.cuda.is_available()` and loads the tensors accordingly, leveraging the compute based on availability.



## Hyperparameters

The following hyperparameters have been considered:

- Learning Rate
- Weight Decay
- Number of Epochs

These have been defined inside the `Utils` class under `get_hyperparameters()` .  The parameters are defined as a dictionary and the function returns a list having the list of parameters respectively. Following this, during runtime, using `itertools.product` to generate a cartesian product of the list of hyperparameters, the training has been done for each set of hyperparameters. 

## Generated output
For given input images original, gray scale and colored reconstructed images <br/>
<img src="https://github.com/dipatidar/Deep-Colorization/blob/main/Results/test_images/input_color.png">
<img src = "https://github.com/dipatidar/Deep-Colorization/blob/main/Results/test_images/output_lab.png">
<img src="https://github.com/dipatidar/Deep-Colorization/blob/main/Results/test_images/output_color.png">




## Contributors

- [Sheela Ippili](https://github.com/sheelaippili)
- [Jimmy Ossa](https://github.com/runninggator)
