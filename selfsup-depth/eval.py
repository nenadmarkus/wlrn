import torch
import numpy as np
import time
import os
import sys
import math
import cv2

from importlib import import_module

def parse_args():
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('modeldef', type=str, help='a script that defines the segmentation network')
	parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')

	return parser.parse_args()

def make_model(modeldef, loadpath):
	init = import_module(modeldef).init
	model = init()
	if loadpath:
		print('* loading pretrained weights from ' + loadpath)
		model.load_state_dict(torch.load(loadpath))
		model.eval()
	else:
		print("* batchnorm is ON for this model")
		model.train()

	model.cuda()

	return model

def _disparity_to_color(I, max_disp=255):
    
    _map = np.array([[0,0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174],
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]]
                   )
    I = np.minimum(I/max_disp, np.ones_like(I))
    
    A = I.transpose()
    num_A = A.shape[0]*A.shape[1]
    
    bins = _map[0:_map.shape[0]-1,3]    
    cbins = np.cumsum(bins)    
    cbins_end = cbins[-1]
    bins = bins/(1.0*cbins_end)
    cbins = cbins[0:len(cbins)-1]/(1.0*cbins_end)
    
    A = A.reshape(1,num_A)            
    B = np.tile(A,(6,1))        
    C = np.tile(np.array(cbins).reshape(-1,1),(1,num_A))
       
    ind = np.minimum(sum(B > C),6)
    bins = 1/bins
    cbins = np.insert(cbins, 0,0)
    
    A = np.multiply(A-cbins[ind], bins[ind])   
    K1 = np.multiply(_map[ind,0:3], np.tile(1-A, (3,1)).T)
    K2 = np.multiply(_map[ind+1,0:3], np.tile(A, (3,1)).T)
    K3 = np.minimum(np.maximum(K1+K2,0),1)
    
    return np.reshape(K3, (I.shape[1],I.shape[0],3)).astype(np.float32).T

def disparity_to_color(disp, max_disp=255):
	m = _disparity_to_color(disp.numpy(), max_disp=max_disp)
	return torch.from_numpy(m).permute(1, 2, 0).float().numpy()

def pad_tensor(t, p):
	b = t.shape[0]
	c = t.shape[1]
	h = (1 + t.shape[2]//p)*p
	w = (1 + t.shape[3]//p)*p
	_t = torch.zeros((b, c, h, w), dtype=t.dtype).to(t.device)
	_t[:, :, 0:t.shape[2], 0:t.shape[3]] = t
	return _t

def calc_disparity(model, img0, img1, max_disp=96, smoothing=None):
	#
	batch = torch.stack((img0, img1))
	batch = batch.cuda()
	_batch = pad_tensor(batch, 32)
	#
	with torch.no_grad():
		featuremaps = model.forward(_batch)
		featuremaps = torch.nn.functional.normalize(featuremaps, p=2, dim=1)
	F0 = featuremaps[0, :, 0:batch.shape[2], 0:batch.shape[3]]
	F1 = featuremaps[1, :, 0:batch.shape[2], 0:batch.shape[3]]
	#
	end_idx = img0.size(2) - 1
	scores = torch.zeros(img0.size(1), img0.size(2), max_disp).cuda()
	for i in range(0, max_disp):
		#
		f0 = F0[:, :, i:end_idx]
		f1 = F1[:, :, 0:end_idx-i]
		#
		scores[:, i:end_idx, i] = torch.sum(torch.mul(f0, f1), 0)
	#
	_, disps = torch.max(scores, 2)
	disps = disps.cpu().byte()
	if smoothing == "median":
		import scipy.signal
		disps = disps.float().numpy()
		disps = torch.from_numpy(scipy.signal.medfilt(disps, 11)).byte()
	elif smoothing == "sgm":
		import sgm.sgm as sgm
		costs = ( (2.0-scores).mul(2048)).cpu().short()
		disps = torch.zeros(scores.size(0), scores.size(1)).short()
		sgm.run(costs, disps)
		disps = disps.byte()
	#
	return disps

def get_bad_pixels(disp, disp_gt, valid_mask):
	disp = disp.float()
	disp_gt = disp_gt.float()

	epe = torch.abs(disp - disp_gt)
	mag = torch.abs(disp_gt)

	outlier_mask = ((epe >= 3.0) & ((epe/mag) >= 0.05))
	outlier_mask[ ~valid_mask ] = False

	color_mask = torch.zeros((3, disp.shape[0], disp.shape[1]), dtype=torch.uint8)
	color_mask[1, :, :][valid_mask] = 255
	color_mask[1, :, :][outlier_mask] = 0
	color_mask[2, :, :][outlier_mask] = 255

	return outlier_mask, color_mask

def compute_kitti_result_for_image_pair(_calc_disparity, folder, name, show=True):
	#
	img0 = torch.from_numpy(cv2.imread(folder+'/image_2/'+name, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)
	img1 = torch.from_numpy(cv2.imread(folder+'/image_3/'+name, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)

	disp = os.path.join(folder, 'disp_occ_0', name)
	if not os.path.exists(disp):
		return None
	else:
		disp = torch.from_numpy(
			cv2.imread(disp, cv2.IMREAD_GRAYSCALE)
		).float()
	#
	disp_calculated = _calc_disparity(img0, img1)

	valid_mask = disp > 0 # "A 0 value indicates an invalid pixel (ie, no ground truth exists, or the estimation algorithm didn't produce an estimate for that pixel)"
	outlier_mask, color_mask = get_bad_pixels(disp_calculated, disp, valid_mask)

	if show:
		cv2.imshow('img0', img0.squeeze(0).numpy())
		cv2.imshow('disp (ground truth, viewed in color)', disparity_to_color(disp))
		disp_calculated[ ~valid_mask ] = 0
		cv2.imshow('disp (calculated, viewed in color)', disparity_to_color(disp_calculated))
		#cv2.imshow('outlier mask (erroneous pixels are white)', (255*outlier_mask.byte()).numpy())
		cv2.imshow('error mask (green=OK, red=error, black=no data)', color_mask.permute(1, 2, 0).numpy())
		if ord('q') == cv2.waitKey(0):
			sys.exit(0)

	return ( outlier_mask.sum() / valid_mask.sum() ).item()

def eval_kitti():
	args = parse_args()

	model = make_model(args.modeldef, args.loadpath)

	def _calc_disparity(img0, img1):
		return calc_disparity(model, img0, img1, smoothing=None)

	nimages = 0
	pctbadpts = 0
	folder = '/home/nenad/Desktop/dev/work/fer/kitti2015/data_scene_flow/training/'
	t = time.time()
	for root, dirs, filenames in os.walk(folder+'/image_2/'):
		for filename in filenames:
			if True:
				p = compute_kitti_result_for_image_pair(_calc_disparity, folder, filename, show=False)
				if p is not None:
					nimages = nimages + 1
					pctbadpts = pctbadpts + p

	print("* elapsed time: %d [sec]" % int(time.time() - t))
	print(nimages, 100*pctbadpts/nimages )

if __name__ == "__main__":
	eval_kitti()
