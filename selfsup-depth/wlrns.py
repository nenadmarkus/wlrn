import torch
import numpy
import importlib
import time
import os
import math

#
# parse command line options
#

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('modeldef', type=str, help='a script that defines the segmentation network')
parser.add_argument('dataloader', type=str, help='a script that loads training and validation samples')
parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')
parser.add_argument('--writepath', type=str, default=None, help='where to write the learned model weights')
parser.add_argument('--learnrate', type=float, default=1e-4, help='RMSprop learning rate')
parser.add_argument('--threshold', type=float, default=0.8, help='WLRN threshold')
parser.add_argument('--dataparallel', action='store_true', default=False, help='wrap the model into a torch.nn.DataParallel module for multi-gpu learning')

args = parser.parse_args()

#
#
#

exec(open(args.modeldef).read())
MODEL = init()
if args.loadpath:
	print('* loading pretrained weights from ' + args.loadpath)
	MODEL.load_state_dict(torch.load(args.loadpath))
MODEL.cuda()

if args.dataparallel:
	print('* using nn.DataParallel')
	MODEL = torch.nn.DataParallel(MODEL)

#
#
#

print('* wlrn threshold set to %f' % args.threshold)
thr = args.threshold
beta = -math.log(1.0/0.99 - 1)/(1.0-thr)

def wlrn_loss_forward(triplet):
	# compute similarities and rescale them to [0, 1]
	AP = torch.mm(triplet[0], triplet[1].t()).add(1).mul(0.5)
	AN = torch.mm(triplet[0], triplet[2].t()).add(1).mul(0.5)
	# kill all scores below `thr`
	AP = torch.sigmoid(AP.add(-thr).mul(beta))
	AN = torch.sigmoid(AN.add(-thr).mul(beta))
	# compute the loss
	return (1 + torch.sum(torch.max(AN, 1)[0]))/(1 + torch.sum(torch.max(AP, 1)[0]))

def loss_forward(featuremaps):
	#
	descs0 = featuremaps[0].permute(1, 2, 0)
	descs1 = featuremaps[1].permute(1, 2, 0)
	#
	losslist = []
	''' v0
	for i in range(0, 16):
		#
		r = numpy.random.randint(0, descs0.size(0))
		#
		a = descs0[r]#.div(torch.norm(descs0[r], 2, 1).view(-1, 1).expand(descs0[r].size())) # .div(...) L2 normalizes descriptors
		p = descs1[r]#.div(torch.norm(descs1[r], 2, 1).view(-1, 1).expand(descs1[r].size()))
		#
		n = []
		for j in range(0, 3):
			stop = False
			while not stop:
				r_neg = numpy.random.randint(0, descs0.size(0))
				if r_neg != r:
					stop = True
			n.append(descs0[r_neg])#.div(torch.norm(descs0[r_neg], 2, 1).view(-1, 1).expand(descs0[r_neg].size())))
		n = torch.cat(n)
		#
		losslist.append( wlrn_loss_forward((a, p, n)) )
	#'''
	#''' v1
	for i in range(0, 16):
		#
		r = numpy.random.randint(16, descs0.size(0)-16)
		#
		a = descs0[r]
		p = descs1[r]
		#
		n = torch.cat([descs0[r-3], descs0[r+3], descs1[r-3], descs1[r+3]])
		#
		losslist.append( wlrn_loss_forward((a, p, n)) )
	#'''
	#
	return sum(losslist)/len(losslist)

#
#
#

print('* data loader: ' + args.dataloader)
exec(open(args.dataloader).read())
loader = get_loader()

optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=args.learnrate)

def train_step(batch):
	#
	avgloss = 0
	optimizer.zero_grad()
	for j in range(0, len(batch)):
		#
		if batch[j][0] is None:
			continue
		#
		featuremaps = MODEL(batch[j].cuda())
		featuremaps = featuremaps.div(torch.norm(featuremaps + 1e-8, 2, 1).unsqueeze(1).expand(featuremaps.size())) # L2 normalize
		loss = loss_forward(featuremaps)
		loss.backward()
		avgloss = avgloss + loss.item()
	optimizer.step()
	avgloss = avgloss/len(batch)
	#
	return avgloss

for epoch in range(0, 256):
	#
	for i, batch in enumerate(loader):
		start = time.time()
		avgloss = train_step(batch)
		print('* batch %d of epoch %d processed in %.4f [s] (average loss: %f)' % (i, epoch, time.time()-start, avgloss))
	#
	if args.writepath and epoch!=0 and epoch%8:
		os.system('mkdir -p ' + args.writepath)
		path = args.writepath + '/' + str(epoch) + '.pth'
		print('* saving model weights to ' + path)
		if args.dataparallel:
			torch.save(MODEL.module.state_dict(), path)
		else:
			torch.save(MODEL.state_dict(), path)
