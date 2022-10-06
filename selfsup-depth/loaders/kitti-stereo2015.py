import os
import numpy
import cv2
import torch
import torch.utils.data

#
def get_load_sample(ROOT='/mnt/sdb1/datasets/kitti2015/data_scene_flow_multiview/training/'):
	#
	samples = []
	for root, dirs, filenames in os.walk(os.path.join(ROOT, 'image_2')):
		for filename in filenames:
			# skip some unsuitable images
			# _10 and _11 are in data_scene_flow.zip / testing/training
			# "exclude neighboring frames (frame 9-12)" in first paragraph of Section "Experimental Setting", Flow2stereo paper
			if any([p in filename for p in ["_09", "_10", "_11", "_12"]]):
				continue
			# add pair to list
			if filename.endswith('.png'):
				samples.append((os.path.join(ROOT, 'image_2', filename), os.path.join(ROOT, 'image_3', filename)))
	#
	def load_sample(index=-1):
		#
		if index<0:
			index = numpy.random.randint(0, len(samples))
		#
		l = cv2.imread(samples[index][0], cv2.IMREAD_COLOR)[22:,:1248,:]
		r = cv2.imread(samples[index][1], cv2.IMREAD_COLOR)[22:,:1248,:]
		#
		return l, r
	#
	return load_sample, len(samples)

def get_loader(ROOT="/mnt/sdb1/datasets/kitti2015/data_scene_flow_multiview/training/"):
	load_sample, nsamples = get_load_sample(ROOT=ROOT)
	class MyDataset(torch.utils.data.Dataset):
		def __init__(self, load_sample, nsamples):
			self.load_sample = load_sample
			self.nsamples = nsamples
		def __len__(self):
			return nsamples
		def __getitem__(self, index):
			l, r = load_sample(index)
			l = torch.from_numpy(l).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
			r = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
			return torch.cat((l, r))
	def collate_fn(batch):
		return batch
	#
	return torch.utils.data.DataLoader(MyDataset(load_sample, nsamples), batch_size=4, collate_fn=collate_fn, shuffle=True)

#
'''
import sys
loader = get_loader(sys.argv[1])
print("* npairs: ", len(loader))
for i, batch in enumerate(loader):
	print(batch[0].shape, batch[0].dtype)
	cv2.imwrite('i0.png', batch[0][0].permute(1, 2, 0).mul(255).numpy())
	cv2.imwrite('i1.png', batch[0][1].permute(1, 2, 0).mul(255).numpy())
	break
#'''
