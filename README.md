# film_Colorization

Dataset has to be [donwloaded](https://davischallenge.org/davis2017/code.html) separately.

## Contents:
[dataset_preparation.ipynb](dataset_preparation.ipynb):
NB for generating the proper training data from DAVIS-videos: RGB (and grey) images, and optical flows

[Train_Demo.ipynb](Train_Demo.ipynb):
NB for training the NNs

[Demo.ipynb](Demo.ipynb):
Render Colored video using a trained Model and weights.



## ToDo's:
- Data:
  - test/train split
  - data samples including >1 frame difference
- Network:
  - UNet, VAE, GAN ?
  - think about Loss function
- Results
  - show performance of fully trained Model (make NB)

Layout I&O:
![NN IO Design](https://github.com/jan-spr/FlowColorization/blob/main/NN%20Diagram.png?raw=true)

## Requirements (so far):
- opencv-python
- scikit-image
- scipy
- pytorch
- torchvision
- matplotlib
- rz-colorization
