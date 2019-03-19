import torch
import imageio

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_model(netG, epoch, model_dir):
	torch.save(netG.state_dict(),'%snetG_epoch_%d.pth' %(model_dir, epoch))

# make gifs
def make_gif(path, num):
	images = []
	for i in range(num):
		file = path + 'fake_samples_epoch_%03d.png' %(i)
		images.append(imageio.imread(file))

	imageio.mimsave(path+'fake_images.gif', images, fps=5)

#make_gif('checkpoints/Image/',19)
