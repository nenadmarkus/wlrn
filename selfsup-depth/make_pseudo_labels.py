import torch
import numpy
import os
import sys
import cv2
import sys
from PIL import Image

from eval import calc_disparity, apply_consistency_filtering, make_model

def load_gray_tensor(path):
	return torch.from_numpy(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).float().div(255.0)

def main(args):
	#
	lr_consistency = True
	#
	model = make_model(args[0], args[1])
	def _calc_disparity(img0, img1):
		d = apply_consistency_filtering(model, img0, img1, 0, 0, None, max_disp=127, filtering=args[2]).float().numpy()
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
		d = _calc_disparity(img0, img1)
		p = sample[0].replace("-l.jpg", "-d.png")

		print(p + ": ", d.shape, d.dtype, d.max(), d.min())

		disp = 256.0*d
		disp[ disp > 2**16-1 ] = 0
		disp = disp.astype(numpy.uint16)
		buff = disp.tobytes()
		img = Image.new("I", disp.T.shape)
		img.frombytes(buff, 'raw', "I;16")
		img.save(p)

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("* args: <modeldef> <params> <filtering> <lr_folder>")
	else:
		main(sys.argv[1:])
