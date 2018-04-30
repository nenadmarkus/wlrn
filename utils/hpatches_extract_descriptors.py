import torch
import torchvision
import sys
import os
import cv2
import numpy

assert len(sys.argv)==6, "* python extract.py <net definition> <net params> <patch size> <HPatches folder> <output folder>"

#
MODELFILE = sys.argv[1]
exec(open(MODELFILE).read())
MODEL = init()
MODEL.load_state_dict( torch.load(sys.argv[2]) )
MODEL.cuda()
MODEL.eval()

MODEL = torch.nn.DataParallel(MODEL)

patchsize = int(sys.argv[3])

#
if not os.path.exists(sys.argv[5]):
	os.makedirs(sys.argv[5])

#
for root, dirs, files in os.walk(sys.argv[4]):
	#
	for f in files:
		if f.endswith('.png'):
			#
			seq = root.split('/')[-1]
			#
			patches = cv2.imread(os.path.join(root, f))
			npatches = int(patches.shape[0]/65)
			patches = cv2.resize(patches, (patchsize, patchsize*npatches))
			patches = torch.from_numpy(patches).view(npatches, patchsize, patchsize, 3).permute(0, 3, 1, 2).cuda().float()
			#torchvision.utils.save_image(patches[400], 'tmp.png')
			#
			descriptors = MODEL.forward(torch.autograd.Variable(patches, volatile=True)).data.cpu().numpy()
			#
			if not os.path.exists(os.path.join(sys.argv[5], seq)):
				os.makedirs(os.path.join(sys.argv[5], seq))
			numpy.savetxt(os.path.join(sys.argv[5], seq, f.split('.')[0]+'.csv'), descriptors, delimiter=',')