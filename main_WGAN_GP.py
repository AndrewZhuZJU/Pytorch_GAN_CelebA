import os
import argparse
from data_loader import get_loader
from args import get_parser
from utils import *
from WGAN import Generator, Discriminator

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import grad
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tensorboardX import summary
from tensorboardX import FileWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(opts):
	opts.checkpoints = opts.checkpoints + 'WGAN-GP/'
	celeba_loader = get_loader(opts.image_dir, opts.dataset, batch_size=opts.batch_size, num_workers=opts.num_workers)
	summary_writer = FileWriter(opts.checkpoints+'/Log/')
	trainer(opts, celeba_loader, summary_writer)


def trainer(opts, dataloader, summary_writer):
	netG = Generator()
	netD = Discriminator()
	netG = nn.DataParallel(netG).to(device)
	netD = nn.DataParallel(netD).to(device)
	print(netG)
	print(netD)
	netG.apply(weights_init)
	netD.apply(weights_init)

	#use RMS, not Adam
	optimizerG = optim.RMSprop(netG.parameters(), lr=opts.lr)
	optimizerD = optim.RMSprop(netD.parameters(), lr=opts.lr)

	#criterion = nn.BCELoss().to(device)

	#gaussion noise for test
	fixed_noise = torch.randn(opts.batch_size, opts.nz, 1, 1, device=device)
	freq = len(dataloader)
	real_labels = torch.FloatTensor(opts.batch_size).fill_(1).to(device)
	fake_labels = torch.FloatTensor(opts.batch_size).fill_(0).to(device)

	for epoch in range(opts.num_epochs):
		for i, (images, labels) in enumerate(dataloader):
			# train D
			netD.zero_grad()
			real_images = images.to(device)
			real_pred = netD(real_images)
			errD_real = -torch.mean(real_pred)
			
			noise = torch.randn(opts.batch_size, opts.nz, 1, 1, device=device)
			fake_images = netG(noise)
			fake_pred = netD(fake_images)
			errD_fake = torch.mean(fake_pred)

			# gradient penalty
			epsilon = torch.rand((opts.batch_size, 1, 1, 1)).to(device)
			x_hat = epsilon * real_images + (1-epsilon) * fake_images
			x_hat_pred = netD(x_hat)
			gradients = grad(outputs=x_hat_pred, inputs=x_hat, grad_outputs=torch.ones(x_hat_pred.size()).to(device),
				create_graph=True, retain_graph=True)[0]
			gradient_penalty = ((gradients.view(gradients.size()[0],-1).norm(2,1)-1)**2).mean()

			errD = errD_real + errD_fake + opts.lambda_gp * gradient_penalty
			errD.backward(retain_graph=True)
			optimizerD.step()

			errG = -torch.mean(fake_pred)
			if(not (i+1)%opts.ncritic):
				# train G
				netG.zero_grad()
				#fake_pred = netD(fake_images)
				errG.backward()
				optimizerG.step()
			print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f GP:%.4f' %(epoch, opts.num_epochs, i, len(dataloader), errD, errG, gradient_penalty))
			if(i % 10==0):
				count = epoch*len(dataloader)+i
				summary_errD = summary.scalar('L_D', errD)
				summary_errD_real = summary.scalar('L_D_real', errD_real)
				summary_errD_fake = summary.scalar('L_D_fake', errD_fake)
				summary_errG = summary.scalar('L_G', errG)
				summary_GP = summary.scalar('GP', gradient_penalty)

				summary_writer.add_summary(summary_errD, count)
				summary_writer.add_summary(summary_errD_real, count)
				summary_writer.add_summary(summary_errD_fake, count)
				summary_writer.add_summary(summary_errG, count)
				summary_writer.add_summary(summary_GP, count)
			if ((i % freq) == 0):
				fake_images = netG(fixed_noise)
				vutils.save_image(fake_images.detach()[0:64],
					'%s/fake_samples_epoch_%03d.png' % (opts.checkpoints+'Image/', epoch), normalize=True)

		if(not (epoch+1)%20):
			save_model(netG, epoch+1, opts.checkpoints+'Model/')


if __name__ == '__main__':
	parse = get_parser()
	opts = parse.parse_args()
	main(opts)