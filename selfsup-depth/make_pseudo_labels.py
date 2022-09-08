import torch
import numpy as np
import time
import os
import sys
import math
import cv2

from eval import calc_disparity, make_model

def writePFM(file, array):
	import os
	assert type(file) is str and type(array) is np.ndarray and os.path.splitext(file)[1] == ".pfm"
	with open(file, 'wb') as f:
		H, W = array.shape
		headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
		for header in headers:
			f.write(str.encode(header))
		array = np.flip(array, axis=0).astype(np.float32)
		f.write(array.tobytes())

def main():
	#
	model = make_model("models.mccnn_large", "models/mccnn_large-e255.pth")
	def _calc_disparity(img0, img1):
		d = calc_disparity(model, img0, img1, filtering="threshold").float().numpy()
		d[:, 0:200] = 0 # ignore the left part of image: matching-based disparities cannot be calculated correctly here
		d[0:100, :] = 0 # ignore the sky, trees
		return d
	#
	#ROOT = '/home/nenad/Desktop/dev/work/fer/kitti2015/data_scene_flow_multiview/training'
	ROOT = '/home/nenad/Desktop/dev/work/fer/kitti2015/data_scene_flow/training/'
	samples = []
	for root, dirs, filenames in os.walk(os.path.join(ROOT, 'image_2')):
		for filename in filenames:
			# skip some unsuitable images
			# _10 and _11 are in data_scene_flow.zip / testing/training
			# "exclude neighboring frames (frame 9-12)" in first paragraph of Section "Experimental Setting", Flow2stereo paper
			if "data_scene_flow_multiview" in ROOT and any([p in filename for p in ["_09", "_10", "_11", "_12"]]):
				continue
			# add pair to list
			if filename.endswith('.png'):
				samples.append((os.path.join(ROOT, 'image_2', filename), os.path.join(ROOT, 'image_3', filename)))
	#
	for sample in samples:
		#
		img0 = torch.from_numpy(cv2.imread(sample[0], cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)
		img1 = torch.from_numpy(cv2.imread(sample[1], cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)
		#
		d = _calc_disparity(img0, img1).astype(np.uint8)
		p = sample[0].split("/")[-1]#.replace(".png", ".pfm")
		p = os.path.join("disp_occ_0", p)
		#writePFM(p, d)
		#print(d.shape, d.dtype, d.max(), d.min())
		print(p)
		cv2.imwrite(p, d)
		#cv2.imshow("...", d)
		#cv2.waitKey(0)

main()
