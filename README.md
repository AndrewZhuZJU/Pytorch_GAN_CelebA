# Pytorch_GAN_CelebA
Pytorch implementation of DCGAN, CDCGAN, LSGAN, WGAN and WGAN-GP for CelebA dataset.

## Usage
### 1. Download the CelebA dataset, and aligned version is used in this repo.
### 2. Clone the repo
```bash
$ git clone https://github.com/AndrewZhuZJU/Pytorch_GAN_CelebA.git
$ cd Pytorch_GAN_CelebA
```
### 3. Training
To train any GAN please use `main_**.py`. For example, 
```bash
$ python main_DCGAN.py
```
Not that all the setting parameters for the modesl are in `args.py`, please change properly.

## Results
* Animation of generated images with fixed noise (or condition labels) during training progress

<table align='center'>
	<tr>
		<td> DCGAN </td>
		<td> LSGAN </td>
	</tr>
	<tr>
		<td><img src='images/dcgan_animation.gif'></td>
		<td><img src='images/lsgan_animation.gif'></td>
	</tr>
</table>

* Generated images in Epoch 40
<table align='center'>
	<tr>
		<td> DCGAN </td>
		<td> LSGAN </td>
	</tr>
	<tr>
		<td><img src='images/dcgan_epoch40.png'></td>
		<td><img src='images/lsgan_epoch40.png'></td>
	</tr>
</table>

## Loss Plot
Coming soon...

## Evaluation
Coming soon...

## Development Environment
* Ubuntu 16.04 LTS
* NVIDIA GTX 1080 Ti
* CUDA 9.0
* pytorch 0.4
* python 2.7
* Others Dependencies: numpy, imageio, torchvision, tensorboard, etc.

## References
Coming soon...