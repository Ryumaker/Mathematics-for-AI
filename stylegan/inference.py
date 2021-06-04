import os
import argparse
import pickle
import math
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator
from train import requires_grad, accumulate, sample_data, adjust_lr

import matplotlib.pyplot as plt


def load_network(ckpt):
	g_running = StyledGenerator(code_size).cuda()
	discriminator = Discriminator(from_rgb_activate=True).cuda()

	ckpt = torch.load(ckpt)
	g_running.load_state_dict(ckpt['g_running'])
	discriminator.load_state_dict(ckpt['discriminator'])

	return g_running, discriminator


def set_random_seed():
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)


def test(discriminator, dataset, generator):
	set_random_seed()

	real_preds, fake_preds = [], []
	real_score, fake_score = 0, 0

	loader = sample_data(dataset, batch_size, resolution)
	for i, (real_index, real_image) in enumerate(loader):
		real_image = real_image.cuda()
		with torch.no_grad():
			real_pred = F.softplus(discriminator(real_image, step=step, alpha=alpha))
			real_preds.append(real_pred)
			real_score += torch.sum(real_pred).item()
	real_preds = torch.cat(real_preds)

	num_samples = 1000


	for i in range(num_samples // batch_size):
		gen_in = torch.randn(batch_size, code_size, device='cuda')
		with torch.no_grad():
			fake_image = generator(gen_in, step=step, alpha=alpha)
			fake_pred = F.softplus(discriminator(fake_image, step=step, alpha=alpha))
			#fake_preds.append(fake_pred)
			#fake_score += torch.sum(fake_pred).item()

	fake_preds = torch.cat(fake_preds)

	real_score /= len(dataset)
	fake_score /= num_samples
	threshold = (real_score + fake_score) / 2

	real_acc = (real_preds > threshold).float().mean().item() * 100
	fake_acc = (fake_preds < threshold).float().mean().item() * 100

	acc = (real_acc + fake_acc) / 2

	return acc, threshold


if __name__ == '__main__':
	code_size = 512
	alpha = 1  # FIX alpha = 1 (no progressive training)

	parser = argparse.ArgumentParser(description='Test generalization of trained discriminators')

	parser.add_argument('--data', type=str, default='test_img', help='dataset 1')
	parser.add_argument('--ckpt', type=str, default='test2_freezeD/best.model', help='model 1')
	parser.add_argument('--image_size', default=256, type=int, help='image size')
	parser.add_argument('--seed', type=int, default=0, help='random seed')

	args = parser.parse_args()

	### load G and D ###

	gen, dis = load_network(f'./checkpoint/{args.ckpt}')


	### load dataset ###

	transform = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
	])

	data = MultiResolutionDataset(f'./dataset/{args.data}_lmdb', transform, resolution=args.image_size)

	step = int(math.log2(args.image_size)) - 2
	resolution = 4 * 2 ** step
	batch_size = 1

	# gen_in = torch.randn(batch_size, code_size, device='cuda')
	# with torch.no_grad():
	# 	fake_image = gen(gen_in, step=step, alpha=alpha)
	#print(fake_image.shape)
	#res = fake_image.cpu()[0].permute(1, 2, 0).numpy()

	#plt.imshow((res*255).astype(np.uint8))
	#plt.show()
	### inference ###
	#acc, threshold = test(dis, data, gen)
	#print(f'acc : {acc} threshold : {threshold}')


