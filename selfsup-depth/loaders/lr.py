import os
import numpy
import cv2
import torch
import torch.utils.data

#
def get_load_sample(ROOT='datasets/LRD'):
	#
	samples = []
	for root, dirs, filenames in os.walk(os.path.join(ROOT)):
		for filename in filenames:
			if filename.endswith("-l.jpg"):
				p = os.path.join(root, filename)
				samples.append((
					p,
					p.replace("-l.jpg", "-r.jpg")
				))
	#
	def load_sample(index=-1):
		#
		if index<0:
			index = numpy.random.randint(0, len(samples))
		#
		l = cv2.imread(samples[index][0], cv2.IMREAD_COLOR)
		r = cv2.imread(samples[index][1], cv2.IMREAD_COLOR)
		#
		if l is None or r is None:
			return None
		#
		return l, r
	#
	return load_sample, len(samples)

def get_loader(usecolor):
	load_sample, nsamples = get_load_sample()
	class MyDataset(torch.utils.data.Dataset):
		def __init__(self, load_sample, nsamples):
			self.load_sample = load_sample
			self.nsamples = nsamples
		def __len__(self):
			return nsamples
		def __getitem__(self, index):
			pair = load_sample(index)
			if pair is None: return None
			l, r = pair
			l = torch.from_numpy(l).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
			r = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
			retval = torch.cat((l, r))
			if usecolor:
				return retval
			else:
				# take just the green channel
				retval = retval[:, 1, :, :]
				return retval.unsqueeze(1).contiguous()
	def collate_fn(batch):
		batch = [s for s in batch if s is not None]
		return batch
	#
	return torch.utils.data.DataLoader(MyDataset(load_sample, nsamples), batch_size=4, collate_fn=collate_fn, shuffle=True)

#
'''
loader = get_loader()
print("* npairs: ", len(loader))
for i, batch in enumerate(loader):
	print(batch[0].shape, batch[0].dtype)
	cv2.imwrite('i0.png', batch[0][0].squeeze().mul(255).numpy())
	cv2.imwrite('i1.png', batch[0][1].squeeze().mul(255).numpy())
	break
#'''
