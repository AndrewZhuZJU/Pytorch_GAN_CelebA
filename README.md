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
### Animation of generated images with fixed noise (or condition labels) during training progress

<table align='center'>
	<tr align='center'>
		<td> DCGAN </td>
		<td> LSGAN </td>
		<td> CGAN(Up:Male, Bottom:Female) </td>
	</tr>
	<tr align='center'>
		<td><img src='Images/dcgan_animation.gif'></td>
		<td><img src='Images/lsgan_animation.gif'></td>
		<td><img src='Images/cgan_animation.gif'></td>
	</tr>
	<tr align='center'>
		<td> WGAN </td>
		<td> WGAN-GP </td>
	</tr>
	<tr align='center'>
		<td><img src='Images/wgan_animation.gif'></td>
		<td><img src='Images/wgan-gp_animation.gif'></td>
	</tr>
</table>

### Generated images in Epoch 40
<table align='center'>
	<tr align=center>
		<td> DCGAN </td>
		<td> LSGAN </td>
		<td> CGAN(Up:Male, Bottom:Female) </td>
	</tr>
	<tr align='center'>
		<td><img src='Images/dcgan_epoch40.png'></td>
		<td><img src='Images/lsgan_epoch40.png'></td>
		<td><img src='Images/cgan_epoch40.png'></td>
	</tr>
	<tr align=center>
		<td> WGAN </td>
		<td> WGAN-GP </td>
	</tr>
	<tr align='center'>
		<td><img src='Images/wgan_epoch40.png'></td>
		<td><img src='Images/wgan-gp_epoch40.png'></td>
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
[1.Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
[2.Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
[3.Wasserstein GAN](https://arxiv.org/abs/1701.07875)
[4.Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
[5.Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
[6.pytorch-generative-model-collections](https://github.com/znxlwm/pytorch-generative-model-collections)
[7.https://github.com/yunjey/StarGAN](https://github.com/yunjey/StarGAN)
