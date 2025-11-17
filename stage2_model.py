import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import timm
# from timm.models.convnext import *
from timm.models.resnet import *

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')

# from config import *
########################################################################################
# unet decoder

class MyCoordDecoderBlock(nn.Module):
	def __init__(
			self,
			in_channel,
			skip_channel,
			out_channel,
			scale=2,
	):
		super().__init__()
		self.scale = scale
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channel + skip_channel+2, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
		)
		self.attention1 = nn.Identity()
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
		)
		self.attention2 = nn.Identity()

	def forward(self, x, skip=None):
		x = F.interpolate(x, scale_factor=self.scale, mode='nearest')  # , align_corners=False) #nearest bilinear
		if skip is not None:
			x = torch.cat([x, skip], dim=1)
			x = self.attention1(x)

		#------------
		b, c, h, w = x.shape
		coordx, coordy = torch.meshgrid(
			torch.linspace(-2, 2, w, dtype=x.dtype, device=x.device),
			torch.linspace(-2, 2, h, dtype=x.dtype, device=x.device),
			indexing='xy'
		)
		coordxy = torch.stack([coordx, coordy], dim=1).reshape(1,2,h,w).repeat(b,1,1,1)
		x = torch.cat([x, coordxy], dim=1)
		#------------

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.attention2(x)
		return x


class MyCoordUnetDecoder(nn.Module):
	def __init__(
			self,
			in_channel,
			skip_channel,
			out_channel,
			scale=[2, 2, 2, 2]
	):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel, ] + out_channel[:-1]
		s_channel = skip_channel
		o_channel = out_channel
		block = [
			MyCoordDecoderBlock(i, s, o, sc)
			for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
		]
		self.block = nn.ModuleList(block)

	def forward(self, feature, skip):
		d = self.center(feature)
		decode = []
		for i, block in enumerate(self.block):
			if 0:
				print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
				print(block.conv1[0])
				print('')
			s = skip[i]
			d = block(d, s)
			decode.append(d)
		last = d
		return last, decode


class UpSampleDeconv(nn.Module):
	def __init__(self, in_ch, mid_ch):
		super().__init__()
		self.up1 = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=2, stride=2)  # s/4 -> s/2
		self.blk = nn.Sequential(
			nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
			nn.BatchNorm2d(mid_ch),
			nn.ReLU(inplace=True),
		)
		# self.up2 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=2, stride=2)  # s/2 -> s/1

	def forward(self, x):
		x = self.up1(x)
		x = self.blk(x)
		# x = self.up2(x)
		return x


def encode_with_convnext(e, x):
	# x 256
	encode = []
	x = e.stem(x)  # 64
	x = e.stages[0](x);
	encode.append(x)  # 64
	x = e.stages[1](x);
	encode.append(x)  # 32
	x = e.stages[2](x);
	encode.append(x)  # 16
	x = e.stages[3](x);
	encode.append(x)  # 8
	return encode


def encode_with_resnet(e, x):
	# x 256

	encode = []
	x = e.conv1(x)
	x = e.bn1(x)
	x = e.act1(x)
	# x = e.maxpool(x)
	x = e.layer1(x);
	encode.append(x)  # 128
	x = e.layer2(x);
	encode.append(x)  # 64
	x = e.layer3(x);
	encode.append(x)  # 32
	x = e.layer4(x);
	encode.append(x)  # 16
	return encode


#####################################################
'''
usage:
- one ECG image has 13 lead signal series 
- input image 1x3xHxW
- truth series CxL (C=13)
- truth pixel 1xCxHxW (C=13)
- truth series mask CxL (C=13) (we have missing values from ecg machine)
- truth pixel mask 1xCxHxW (C=13)

coord = torch.arange(H)
y0 = torch.tensor([13 y0 values)] #zero mv line
pixel_to_mV = 1/(2*39.348837209302324)  


probability = UNET(rectified ECG image)  #softmax per pixel for 13+1 class, 1 is for background
#probability is 1x(C+1)xHxW

msel_loss, snr_loss = regression_loss(
	probability, coord, truth_series, truth_series_mask,
	y0,  
	pixel_to_mV
)
js_loss = regularize_loss(probability, truth_pixel, truth_pixel_mask)

'''


# direct regression loss from mask probability
def normalize_prob(x, dim=2, eps=1e-8):
	# Ensure proper distributions that sum to 1 along `dim`
	s = x.sum(dim=dim, keepdim=True)
	return x / (s + eps)


def js_divergence(p, q, dim=-1, eps=1e-8):
	"""
	p, q: probability tensors (already normalized along `dim`)
	Returns JS divergence reduced over `dim` (keeps other dims).
	"""
	m = 0.5 * (p + q)
	kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=dim)
	kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=dim)
	return 0.5 * (kl_pm + kl_qm)


# Jensen-Shannon divergence
def regularize_loss(probability, truth):
	EPS = 1e-6

	B, C, H, W = probability.shape

	p = normalize_prob(probability, dim=2, eps=EPS)
	q = normalize_prob(truth, dim=2, eps=EPS)
	js = js_divergence(p, q, dim=2, eps=EPS)  # B,C,W summed over `dim`

	#valid = mask.any(dim=2).to(js.dtype)  # (B,C,W)
	#js_loss = (js * valid).sum() / (valid.sum() + EPS)
	js_loss = js.mean()
	return js_loss


# ---------------------------------------------------------



def F_snr(
	predict, truth
):
	eps = 1e-7
	signal = (truth ** 2).sum()
	noise = ((predict - truth) ** 2).sum()
	snr = signal / (noise + eps)
	snr_db = 10 * torch.log10(snr + eps)
	snr_loss = -snr_db.mean()
	return snr_loss

# Right now SNR is per-sample scalar; good for a metric but not a loss.
# total_loss = msel_loss + 0.01 * snr_loss


def prob_to_series(p, L=None):
	# p: (B,1,H,W)  ->  mu_y: (B,1,W) in pixel units [0..H-1]
	B, _, H, W = p.shape
	y = torch.linspace(0, H - 1, H, device=p.device).view(1, 1, H, 1)
	s = (p * y).sum(dim=2, keepdim=True)  # (B,1,1,W)
	series = s.squeeze(2)  # (B,1,W)
	if L is not None:
		series = F.interpolate(series, size=L, mode='linear', align_corners=False)

	return series


def prob_to_series_by_max(p, L=None):
	B, _, H, W = p.shape
	series = p.argmax(dim=2, keepdim=False)
	if L is not None:
		series = F.interpolate(series.float(), size=L, mode='linear', align_corners=False)
	return series

def prob_to_series_by_max1(p, L=None):
	B, _, H, W = p.shape
	#series = p.argmax(dim=2, keepdim=False)
	p = p**5
	y = torch.linspace(0, H - 1, H, device=p.device).view(1, 1, H, 1)
	series = (p * y).sum(dim=2, keepdim=False)/ (p).sum(dim=2, keepdim=False) # (B,1,1,W)

	if L is not None:
		series = F.interpolate(series.float(), size=L, mode='linear', align_corners=False)
	return series

###############################################################################3
NUM_LEAD = 13

#
class Net(nn.Module):
	def __init__(self,pretrained=True):
		super(Net, self).__init__()
		self.max_shift_px = 3

		self.output_type = ['infer', 'loss']
		self.register_buffer('D', torch.tensor(0))
		self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
		self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

		# arch = 'convnext_tiny.fb_in22k' #
		# arch = 'convnext_small.fb_in22k'
		#arch = 'resnet10t.c3_in1k'
		# arch = 'resnet18d.ra4_e3600_r224_in1k'
		arch = 'resnet34.a3_in1k'

		encoder_dim = {
			'resnet18d.ra4_e3600_r224_in1k': [64, 128, 256, 512],
			'resnet10t.c3_in1k': [64, 128, 256, 512],
			'resnet34.a3_in1k': [64, 128, 256, 512],
			'convnext_tiny_in22k': [96, 192, 384, 768],
			'convnext_tiny.fb_in22k': [96, 192, 384, 768],
			'convnext_small.fb_in22k': [96, 192, 384, 768],  # 96, 192, 384, 768
			'convnext_base.fb_in22k': [128, 256, 512, 1024],  # 96, 192, 384, 768
			'resnet50d': [64, 256, 512, 1024, 2048, ],
		}[arch]
		decoder_dim = [256, 128, 64, 32]
		# self.upby2 = UpSampleDeconv(decoder_dim[-1],decoder_dim[-1])

		self.encoder = timm.create_model(
			model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
		)

		self.decoder = MyCoordUnetDecoder(
			in_channel=encoder_dim[-1],
			skip_channel=encoder_dim[:-1][::-1] + [0],
			out_channel=decoder_dim,
			scale=[2, 2, 2, 2]
		)
		self.pixel = nn.Conv2d(decoder_dim[-1]+1, 4, 1)

	# self.dc_pluse = nn.Conv2d(decoder_dim[-1], NUM_MARKER + 1, kernel_size=1)  #softmax

	# todo image level grade ???

	def forward(self, batch, L=None):
		device = self.D.device
		# if 'loss' in self.output_type:
		# 	L = batch['series'].shape[-1]

		image = batch['image'].to(device)
		B, _3_, H, W = image.shape
		x = image.float() / 255
		x = (x - self.mean) / self.std

		coordy = torch.arange(H, device=device).reshape(1, 1, H, 1).repeat(B, 1, 1, W)
		coordx = torch.arange(W, device=device).reshape(1, 1, 1, W).repeat(B, 1, H, 1)
		coordy = coordy/(H-1)*2-1
		coordx = coordy/(W-1)*2-1

		# unet ------------------------------------
		e = self.encoder
		encode = encode_with_resnet(e, x)
		last, decode = self.decoder(
			feature=encode[-1], skip=encode[:-1][::-1] + [None]
		)

		# head
		B,_,h,w = last.shape
		last = torch.cat([last, coordy], dim=1)

		pixel = self.pixel(last)
		# p = (6.0 * pixel).softmax(dim=2)
		# p = p.clamp_min(1e-12).pow(2.0);
		# prob = p / (p.sum(dim=2, keepdim=True) + 1e-12)
		#p = batch['pixel'].to(device)  +0*p



		#series = prob_to_series(prob, L)
		output = {}
		if 'loss' in self.output_type:
			# output['pixel_loss'] = regularize_loss(
			# 	prob,
			# 	truth=batch['pixel'].to(device),
			# )
			# output['pixel_loss']=F.cross_entropy(
			# 	pixel.squeeze(1),
			# 	batch['pixel'].to(device).max(2)[1].squeeze(1),
			# )
			output['pixel_loss'] = F.binary_cross_entropy_with_logits(
				pixel,
				batch['pixel'].to(device),
				pos_weight=torch.tensor([10]).to(device),
			)
			#---
			# output['mse_loss'] = F.mse_loss(
			# 	series, batch['series'].to(device)
			# )
			# output['snr_loss'] = F_snr(
			# 	series, batch['series'].to(device)
			# )

		if 'infer' in self.output_type:
			output['pixel'  ] = torch.sigmoid(pixel)


		return output


# todo: should i use 3x3 conv feature at zero scale? add another encode feature?


def run_check_net():
	H, W = 320, 320
	batch_size = 4


	batch = { #dummy data
		'image':  torch.from_numpy(np.random.randint(0, 256, (batch_size, 3, H, W))).byte(),
		'pixel':  torch.from_numpy(np.random.choice(13 + 1, (batch_size, 4, H, W))).float(),
	}


	net = Net(pretrained=True).cuda()
	# print(net)

	with torch.no_grad():
		with torch.amp.autocast('cuda'):
			output = net(batch)
	# ---

	print('batch')
	for k, v in batch.items():
		if k in ['sampling_length','pixel_to_mv']:
			print(f'{k:>32} : {v} ')
		else:
			print(f'{k:>32} : {v.shape} ')

	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print(f'{k:>32} : {v.shape} ')
	print('loss')
	for k, v in output.items():
		if 'loss' in k:
			print(f'{k:>32} : {v.item()} ')

# main #################################################################
if __name__ == '__main__':
	run_check_net()