import torch
import torch.nn.functional as F

import cv2
import numpy as np
import os
import pandas as pd

##################################################################3
# helper


############################################################################################
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
print(THIS_DIR)


#############################################################################################
# visualisation

def show_image(
		image,
		name='image',
		mode=cv2.WINDOW_AUTOSIZE,  # WINDOW_NORMAL
		resize=None
):
	cv2.namedWindow(name, mode)
	if image.ndim == 3:
		cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
	if image.ndim == 2:
		cv2.imshow(name, image)
	if resize is not None:
		H, W = image.shape[:2]
		cv2.resizeWindow(name, int(resize * W), int(resize * H))


# more
def draw_lead_pixel(image, pixel):
	overlay = image // 2
	overlay = 255 - (255 - overlay) * (1 - pixel[0][..., np.newaxis] * [[[1, 0, 0]]])
	overlay = 255 - (255 - overlay) * (1 - pixel[1][..., np.newaxis] * [[[0, 1, 0]]])
	overlay = 255 - (255 - overlay) * (1 - pixel[2][..., np.newaxis] * [[[0, 0, 1]]])
	overlay = 255 - (255 - overlay) * (1 - pixel[3][..., np.newaxis] * [[[1, 1, 0]]])
	overlay = overlay.astype(np.uint8)
	return overlay


############################################################################################

def load_net(net, checkpoint_file):
	f = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
	state_dict = f['state_dict']
	print(net.load_state_dict(state_dict, strict=False))

	net.eval()
	net.output_type = ['infer']
	return net


# series processing---------------------------------------
def np_snr(
	predict, truth
):
	eps = 1e-7
	signal = (truth ** 2).sum()
	noise = ((predict - truth) ** 2).sum()
	snr = signal / (noise + eps)
	snr_db = 10 * np.log10(snr + eps)
	snr_loss = -snr_db.mean()
	return snr_loss


def split_to_lead(series, split_length):
	LEAD = [
		'I', 'aVR', 'V1', 'V4',
		'II', 'aVL', 'V2', 'V5',
		'III', 'aVF', 'V3', 'V6',
	]
	index= np.cumsum(split_length)[:-1]

	lead = {}
	for i in range(3):
		split= np.split(series[i], index)
		for (k, s) in zip(LEAD[i], split):
			lead[k]=s
	lead['II-rhythm'] = series[3]
	return lead


def pixel_to_series(pixel, zero_mv, length):
	_, H, W = pixel.shape
	assert (H == 1696)

	series = []
	for j in [0, 1, 2, 3]:
		# j=0
		# p = pixel[j]
		p = pixel[j]

		# find max (upper envolope)
		amax = p.argmax(0)
		amin = H - 1 - p[::-1].argmax(0)  # crop H=1696
		mask = amax >= zero_mv[j]

		s1 = mask * amax + (1 - mask) * amin
		s  = amax

		# find missing
		miss = ((p > 0.1).sum(0) == 0)  # * 100 + zero_mv[j]
		s[miss] = zero_mv[j]
		series.append(s)
	series = np.stack(series).astype(np.float32)

	if length is not None:
		if length!=W:
			series = torch.from_numpy(series).unsqueeze(1)
			series = F.interpolate(series, size=length, mode='linear', align_corners=False)
			series = series.squeeze(1).data.cpu().numpy()

	return series

def filter_series_by_limits(
	series,
	limits=[
		[-2,2,], [-2,2,], [-4,4,],[-4,4,],
	]
):
	_, L = series.shape
	series[3] = np.clip(series[3], -1, 1) #'II-rhythm'
	for j in [0,1,2]:
		for i in range(4):
			i0=i*(L//4)
			i1=(i+1)*(L//4)
			series[j:i0:i1] = np.clip(series[j:i0:i1], *limits[i])

	return series

def read_truth_series(sample_id,KAGGLE_DIR):
	image_id = sample_id.split('-')[0]
	truth_df = pd.read_csv(f'{KAGGLE_DIR}/train/{image_id}/{image_id}.csv')
	truth_df['II-rhythm'] = truth_df['II']
	truth_df.loc[truth_df['I'].isna(), 'II'] = np.nan
	truth_df.fillna(0, inplace=True)

	series0 = (truth_df['I'] + truth_df['aVR'] + truth_df['V1'] + truth_df['V4']).values
	series1 = (truth_df['II'] + truth_df['aVL'] + truth_df['V2'] + truth_df['V5']).values
	series2 = (truth_df['III'] + truth_df['aVF'] + truth_df['V3'] + truth_df['V6']).values
	series3 = (truth_df['II-rhythm']).values
	truth_df['series0']=series0
	truth_df['series1']=series1
	truth_df['series2']=series2
	truth_df['series3']=series3
	return truth_df