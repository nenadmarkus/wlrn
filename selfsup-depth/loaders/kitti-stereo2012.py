import os
import numpy
import cv2
import torch
import torch.utils.data

#
def get_loader():
	#
	ROOT = 'data/kitti-stereo2012/training/'
	samples = []
	for root, dirs, filenames in os.walk(os.path.join(ROOT, 'image_0')):
		for filename in filenames:
			if filename.endswith('.png'):
				samples.append((os.path.join(ROOT, 'image_0', filename), os.path.join(ROOT, 'image_1', filename)))
	#
	def load_sample(index=-1):
		#
		if index<0:
			index = numpy.random.randint(0, len(samples))
		#
		img0 = torch.from_numpy(cv2.imread(samples[index][0], cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0).float().div(255.0)
		img1 = torch.from_numpy(cv2.imread(samples[index][1], cv2.IMREAD_GRAYSCALE)).unsqueeze(0).unsqueeze(0).float().div(255.0)
		#
		return torch.cat([img0, img1])
	#
	nsamples = len(samples)
	class MyDataset(torch.utils.data.Dataset):
		def __init__(self, load_sample, nsamples):
			self.load_sample = load_sample
			self.nsamples = nsamples
		def __len__(self):
			return nsamples
		def __getitem__(self, index):
			return load_sample(index)
	def collate_fn(batch):
		return batch
	#
	return torch.utils.data.DataLoader(MyDataset(load_sample, nsamples), batch_size=4, collate_fn=collate_fn, shuffle=True)

#
'''
loader = get_loader()
for i, batch in enumerate(loader):
	cv2.imwrite('i0.png', batch[0][0].squeeze().mul(255).numpy())
	cv2.imwrite('i1.png', batch[0][1].squeeze().mul(255).numpy())
	break
#'''