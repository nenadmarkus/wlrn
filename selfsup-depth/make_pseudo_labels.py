import torch
import numpy as np
import time
import os
import sys
import math
import cv2
import sys

from eval import calc_disparity, apply_left_right_consistency, make_model

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

def load_gray_tensor(path):
	return torch.from_numpy(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)

def main(args):
	#
	lr_consistency = True
	#
	model = make_model(args[0], args[1])
	def _calc_disparity(img0, img1):
		if not lr_consistency:
			d = calc_disparity(model, img0, img1, filtering=args[2]).float().numpy()
			d[:, 0:100] = 0 # ignore the left part of image: matching-based disparities cannot be calculated correctly here
			d[0:100, :] = 0 # ignore the sky, trees
		else:
			d = apply_left_right_consistency(model, img0, img1, 1, filtering=args[2]).float().numpy()
			d[0:50, :] = 0 # ignore the sky, trees
		return d
	#
	samples = []
	for root, dirnames, filenames in os.walk(args[3]):
		for filename in filenames:
			if filename.endswith("-l.jpg"):
				samples.append((
					os.path.join(root, filename),
					os.path.join(root, filename).replace("-l.jpg", "-r.jpg")
				))
	#
	for sample in samples:
		#
		img0 = load_gray_tensor(sample[0])
		img1 = load_gray_tensor(sample[1])
		#
		d = _calc_disparity(img0, img1).astype(np.uint8)
		p = sample[0].replace("-l.jpg", "-d.png")
		#writePFM(p, d)
		#print(d.shape, d.dtype, d.max(), d.min())
		print(p)
		cv2.imwrite(p, d)
		#cv2.imshow("...", d)
		#cv2.waitKey(0)

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("* args: <modeldef> <params> <filtering> <lr_folder>")
	else:
		main(sys.argv[1:])
