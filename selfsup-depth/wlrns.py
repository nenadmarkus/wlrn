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

def compute_matrix_entropy_loss(triplet, temp=10):
	# use only anchor and positive
	a = triplet[0]
	p = triplet[1]
	# similarity and probability matrices
	S = torch.matmul(a, p.t())
	P = torch.softmax(temp*S, dim=1)
	#print(P[0].max().item(), P[0].min().item(), P[0].mean().item())
	# compute the average entropy (per row)
	H = - torch.mul(P, torch.log(P))
	H = H.sum() / a.shape[0]
	# we want to minimize entropy (i.e., we want the distribution to be spiky)
	#print(H.item())
	return H

def loss_forward(featuremaps):

	descs0 = featuremaps[0].permute(1, 2, 0)
	descs1 = featuremaps[1].permute(1, 2, 0)

	losslist = []

	for i in range(0, 16):
		#
		r = numpy.random.randint(16, descs0.size(0)-16)
		#
		a = descs0[r]
		p = descs1[r]
		#
		n = torch.cat([descs0[r-3], descs0[r+3], descs1[r-3], descs1[r+3]])
		#
		losslist.append( compute_matrix_entropy_loss((a, p, n)) )

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
	#if args.writepath and epoch!=0 and 0==epoch%8:
	if args.writepath and epoch!=0:
		os.system('mkdir -p ' + args.writepath)
		path = args.writepath + '/' + str(epoch) + '.pth'
		print('* saving model weights to ' + path)
		if args.dataparallel:
			torch.save(MODEL.module.state_dict(), path)
		else:
			torch.save(MODEL.state_dict(), path)
