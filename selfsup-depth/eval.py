import torch
import numpy as np
import time
import os
import sys
import math
import cv2

usecuda = torch.cuda.is_available()
usecolor = False

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

def calc_disparity(model, img0, img1, max_disp=96, filtering=None):
	#
	batch = torch.stack((img0, img1))
	if usecuda:
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
	scores = torch.zeros(img0.size(1), img0.size(2), max_disp)
	if usecuda:
		scores = scores.cuda()
	for i in range(0, max_disp):
		#
		f0 = F0[:, :, i:end_idx]
		f1 = F1[:, :, 0:end_idx-i]
		#
		scores[:, i:end_idx, i] = torch.sum(torch.mul(f0, f1), 0)
	#
	sims, disps = torch.max(scores, 2)
	disps = disps.cpu().byte()
	if filtering == "none":
		pass
	elif filtering == "threshold":
		disps[ sims < 0.75 ] = 0
	elif filtering == "median":
		disps = cv2.medianBlur(disps.numpy(), 17)
		disps = torch.from_numpy(disps).byte()
	elif filtering == "sgm":
		import sgm
		dists = 0.5*(1.0 - scores)
		costvol = (2048*dists).cpu().int().numpy()
		disps = sgm.run(costvol, max_disp)
		disps = torch.from_numpy(disps)

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

def compute_kitti_result_for_image_pair(_calc_disparity, folder, name, show=True, vizdir="vizdir", ignore_calc_zeros=False):
	#
	if usecolor:
		img0 = torch.from_numpy(cv2.imread(folder+'/image_2/'+name, cv2.IMREAD_COLOR)).permute(2, 0, 1).float().div(255.0)
		img1 = torch.from_numpy(cv2.imread(folder+'/image_3/'+name, cv2.IMREAD_COLOR)).permute(2, 0, 1).float().div(255.0)
	else:
		img0 = torch.from_numpy(cv2.imread(folder+'/image_2/'+name, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)
		img1 = torch.from_numpy(cv2.imread(folder+'/image_3/'+name, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)

	disp = os.path.join(folder, 'disp_occ_0', name)
	if not os.path.exists(disp):
		return None
	else:
		disp = torch.from_numpy(
			cv2.imread(disp, cv2.IMREAD_GRAYSCALE)
		).float()
	#disp[:, 0:200] = 0
	#
	disp_calculated = _calc_disparity(img0, img1)

	# for gound truth: "A 0 value indicates an invalid pixel (ie, no ground truth exists, or the estimation algorithm didn't produce an estimate for that pixel)"
	# for predicted: we can doscard some values based on low matching threshold
	if ignore_calc_zeros:
		valid_mask = (disp > 0) & (disp_calculated > 0)
	else:
		valid_mask = (disp > 0)
	outlier_mask, color_mask = get_bad_pixels(disp_calculated, disp, valid_mask)

	if vizdir is not None:
		os.system("mkdir -p '%s'" % vizdir)
		left = cv2.resize(cv2.imread(folder+'/image_2/'+name, cv2.IMREAD_COLOR), None, fx=0.5, fy=0.5)
		right = cv2.resize(cv2.imread(folder+'/image_3/'+name, cv2.IMREAD_COLOR), None, fx=0.5, fy=0.5)
		cm = color_mask.permute(1, 2, 0).numpy()
		viz = np.zeros((left.shape[0]+disp.shape[0]+disp_calculated.shape[0]+cm.shape[0], disp.shape[1], 3), dtype=np.uint8)
		viz[:left.shape[0], :left.shape[1], :] = left
		viz[:right.shape[0], left.shape[1]:(left.shape[1]+right.shape[1]), :] = right
		viz[right.shape[0]:(right.shape[0]+disp.shape[0]), :, :] = 255*disparity_to_color(disp)
		d2c = 255*disparity_to_color(disp_calculated)
		viz[(right.shape[0]+disp.shape[0]):(right.shape[0]+disp.shape[0]+d2c.shape[0]), :, :] = d2c
		viz[(right.shape[0]+disp.shape[0]+d2c.shape[0]):, :, :] = cm
		cv2.imwrite(os.path.join(vizdir, name).replace(".png", ".jpg"), viz)

	if show:
		disp_calculated[ ~valid_mask ] = 0
		cv2.imshow('disp (calculated, viewed in color)', disparity_to_color(disp_calculated))
		cv2.imshow('error mask (green=OK, red=error, black=no data)', color_mask.permute(1, 2, 0).numpy())
		if ord('q') == cv2.waitKey(0):
			sys.exit(0)

	if valid_mask.sum() == 0:
		return None
	else:
		return ( outlier_mask.sum() / valid_mask.sum() ).item()

def eval_kitti2015(model, folder, filtering, vizdir):
	def _calc_disparity(img0, img1):
		return calc_disparity(model, img0, img1, filtering=filtering)

	nimages = 0
	pctbadpts = 0
	t = time.time()
	for root, dirs, filenames in os.walk(folder+'/image_2/'):
		for filename in filenames:
			if True:
				p = compute_kitti_result_for_image_pair(_calc_disparity, folder, filename, show=False, vizdir=vizdir, ignore_calc_zeros=filtering=="threshold")
				if p is not None:
					print("%s        |        %.1f" % (filename, 100*p))
					nimages = nimages + 1
					pctbadpts = pctbadpts + p

	print("* elapsed time (eval): %d [sec]" % int(time.time() - t))
	#print(nimages, 100*pctbadpts/nimages )
	return 100*pctbadpts/nimages

#
#
#

from importlib import import_module

def make_model(modeldef, loadpath):
	global usecolor
	init = import_module(modeldef).init
	model = init(1)
	if loadpath:
		print('* loading pretrained weights from ' + loadpath)
		try:
			model.load_state_dict(torch.load(loadpath, map_location=torch.device("cpu")))
			usecolor = False
		except:
			model = init(3)
			model.load_state_dict(torch.load(loadpath, map_location=torch.device("cpu")))
			usecolor = True
		model.eval()
		print("* color: " + str(usecolor))
	else:
		print("* batchnorm is ON for this model")
		model.train()

	if usecuda:
		model.cuda()

	return model

def parse_args():
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('modeldef', type=str, help='a script that defines the segmentation network')
	parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')
	parser.add_argument('--kittipath', type=str, default="datasets/kitti2015/data_scene_flow/training/", help='path to KITTI data')
	parser.add_argument('--filtering', type=str, default="median", help='filtering applied to the raw disparity map')
	parser.add_argument('--vizdir', type=str, default=None, help='directory where to store visualizations')

	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	model = make_model(args.modeldef, args.loadpath)
	p = eval_kitti2015(model, args.kittipath, args.filtering, args.vizdir)
	print("* bad points: %.2f%%" % p)
