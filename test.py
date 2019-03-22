import os
import argparse
from data_loader import get_loader
from args import get_parser
from utils import *
from DCGAN import Generator, Discriminator

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tensorboardX import summary
from tensorboardX import FileWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(opts):
	opts.checkpoints = opts.checkpoints + 'DCGAN/'
	#celeba_loader = get_loader(opts.image_dir, opts.dataset, batch_size=opts.batch_size, num_workers=opts.num_workers)
	test(opts)


def test(opts):
	netG = Generator()
	netG = nn.DataParallel(netG).to(device)
	#netD = nn.DataParallel(netD).to(device)
	netG.load_state_dict(torch.load(opts.checkpoints+'Model/netG_epoch_20.pth'))
	netG.eval()

	count = 0
	while(True):
		noise = torch.randn(opts.batch_size, opts.nz, 1, 1, device=device)
		fake_images = netG(noise)
		for i in range(opts.batch_size):
			vutils.save_image(fake_images.detach()[i],
					'%s%04d.png' % ('Generated/0/', count), normalize=True)
			count += 1
		if(count > 30000):
			break
		print(count)


if __name__ == '__main__':
	parse = get_parser()
	opts = parse.parse_args()
	main(opts)