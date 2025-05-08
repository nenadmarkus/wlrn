import torch
import numpy as np
import sys
import cv2
import sys

from eval import calc_disparity, apply_consistency_filtering, make_model, _disparity_to_color

def get_gray_tensor(img):
	return torch.from_numpy(img).unsqueeze(0).float().div(255.0)

def calculate_dispmap(l, r, model, filtering="none", lr_consistency = True):
	def _calc_disparity(img0, img1):
		if not lr_consistency:
			d = calc_disparity(model, img0, img1, filtering=filtering, max_disp=255).float().numpy()
		else:
			d = apply_consistency_filtering(model, img0, img1, 0, 0, None, filtering=filtering, max_disp=255).float().numpy()
		return d
	tn_l = get_gray_tensor(l)
	tn_r = get_gray_tensor(r)

	d = _calc_disparity(tn_l, tn_r).astype(np.uint8)
	return d

def main(args):
	model = make_model(args[0], args[1])

	#
	if len(args) == 4:
		lr_cat = cv2.imread(args[3], cv2.IMREAD_GRAYSCALE)
		lr_cat = cv2.resize(lr_cat, None, fx=0.25, fy=0.25)
		l = lr_cat[:, 0:(lr_cat.shape[1]//2)]
		r = lr_cat[:, (lr_cat.shape[1]//2):]
	else:
		l = cv2.imread(args[3], cv2.IMREAD_GRAYSCALE)
		r = cv2.imread(args[4], cv2.IMREAD_GRAYSCALE)

	d = calculate_dispmap(l, r, model, lr_consistency=True)

	print(d.shape, np.max(d), np.min(d))

	c = _disparity_to_color(d, max_disp=np.max(d)).transpose((1, 2, 0))
	cv2.imshow("dispmap", c)
	cv2.waitKey(0)

#
#
#

# python run_single.py models.mccnn_large models/mccnn_large.pth none 2245-l.jpg 2245-r.jpg

if __name__ == "__main__":
	print("* args: <modeldef> <params> <filtering> <l> <r>")
	main(sys.argv[1:])
