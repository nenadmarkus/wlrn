import torch
import torch.nn as nn
import numpy as np
import importlib
import time
import os
import math
import cv2

#
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('modeldef', type=str, help='a script that defines the segmentation network')
parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')

args = parser.parse_args()

#
exec(open(args.modeldef).read())
MODEL = init()
if args.loadpath:
	print('* loading pretrained weights from ' + args.loadpath)
	MODEL.load_state_dict(torch.load(args.loadpath))
	MODEL.eval()
else:
	print("* batchnorm is ON for this model")
	MODEL.train()

MODEL.cuda()

#
max_disp = 96

#
def disparity_to_color(I, max_disp):
    
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

def calc_disparity(img0, img1):
	#
	batch = torch.cat([img0.unsqueeze(0), img1.unsqueeze(0)]).cuda()
	#
	with torch.no_grad():
		featuremaps = MODEL(batch)
	featuremaps = featuremaps.div(torch.norm(featuremaps, 2, 1).unsqueeze(1).expand(featuremaps.size())) # L2 normalize
	#
	end_idx = img0.size(2) - 1
	scores = torch.zeros(img0.size(1), img0.size(2), max_disp).cuda()
	for i in range(0, max_disp):
		#
		f0 = featuremaps[0, :, :, i:end_idx]
		f1 = featuremaps[1, :, :, 0:end_idx-i]
		#
		scores[:, i:end_idx, i] = torch.sum(torch.mul(f0, f1), 0)
	#
	_, disps = torch.max(scores, 2)
	disps = disps.cpu().byte()
	'''
	import scipy.signal
	disps = disps.float().numpy()
	disps = torch.from_numpy(scipy.signal.medfilt(disps, 11)).byte()
	#'''
	'''
	import sgm.sgm as sgm
	costs = ( (2.0-scores).mul(2048)).cpu().short()
	disps = torch.zeros(scores.size(0), scores.size(1)).short()
	sgm.run(costs, disps)
	disps = disps.byte()
	#'''
	#
	return disps

#
def count_bad_points(disp, disp_calculated, mask, thr, img0=None):
	#
	delta = (disp_calculated.float() - disp.float()).abs()
	masked = torch.mul(delta, mask)
	lethr = delta.le(thr).float() - mask.eq(0).float()
	#
	disp_calc_color = disparity_to_color(disp_calculated.numpy(), max_disp - 1)
	disp_calc_color = torch.from_numpy(disp_calc_color).permute(1, 2, 0).float().numpy()

	dicp_color = disparity_to_color(disp.numpy(), max_disp - 1)
	dicp_color = torch.from_numpy(dicp_color).permute(1, 2, 0).float().numpy()

	if show:
		cv2.imshow('img0', img0.squeeze(0).numpy())
		cv2.imshow('disp (ground truth, viewed in color)', dicp_color)
		cv2.imshow('disp (calculated, viewed in color)', disp_calc_color)
		cv2.imshow('accurately predicted points', lethr.numpy())
		cv2.waitKey(0)

	return 1.0 - 1.0*lethr.sum()/mask.sum()

#
def compute_kitti_result_for_image_pair(folder, name, show=True):
	#
	img0 = torch.from_numpy(cv2.imread(folder+'/image_2/'+name, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)
	img1 = torch.from_numpy(cv2.imread(folder+'/image_3/'+name, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)
	disp = cv2.imread(folder+'/disp_noc_0/'+name, cv2.IMREAD_GRAYSCALE)
	if disp is None:
		return None
	disp = torch.from_numpy(disp)
	#
	disp_calculated = calc_disparity(img0, img1)
	mask = 1.0 - disp.eq(0).float()
	disp_calculated = torch.mul(disp_calculated.float(), mask).byte()
	#
	if show:
		return count_bad_points(disp, disp_calculated, mask, 2, img0)
	else:
		return count_bad_points(disp, disp_calculated, mask, 2, None)

'''
folder = '/home/nenad/Desktop/dev/work/fer/kitti2015/data_scene_flow/training/'
imgname = '000000_10.png'
print( 100*compute_kitti_result_for_image_pair(folder, imgname) )
#'''

#'''
nimages = 0
pctbadpts = 0
folder = '/home/nenad/Desktop/dev/work/fer/kitti2015/data_scene_flow/training/'
for root, dirs, filenames in os.walk(folder+'/image_2/'):
	for filename in filenames:
		if True:
			#
			p = compute_kitti_result_for_image_pair(folder, filename, show=False)
			if p is not None:
				nimages = nimages + 1
				pctbadpts = pctbadpts + p
#
print(nimages, 100*pctbadpts/nimages )
#'''
