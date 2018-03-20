#
import torch
import torch.nn.functional as F

import sys
import math
import time
import random

#
# parse command line options
#

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('modeldef', type=str, help='a script that define the descriptor-extractor model')
parser.add_argument('tripletgen', type=str, help='a script that generates training and validation triplets')
parser.add_argument('--writepath', type=str, default=None, help='where to write the learned model weights')
parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')
parser.add_argument('--learnrate', type=float, default=1e-4, help='RMSprop learning rate')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')
parser.add_argument('--dataparallel', action='store_true', default=False, help='wrap the model into a torch.nn.DataParallel module for multi-gpu learning')

args = parser.parse_args()

#
# model
#

exec(open(args.modeldef).read())
MODEL = init()
if args.loadpath:
	print('* loading pretrained weights from ' + args.loadpath)
	MODEL.load_state_dict(torch.load(args.loadpath))
if args.dataparallel:
	print('* using nn.DataParallel')
	MODEL = torch.nn.DataParallel(MODEL)
MODEL.cuda()

def model_forward(triplet):
	#
	return [
		MODEL.forward(triplet[0]),
		MODEL.forward(triplet[1]),
		MODEL.forward(triplet[2])
	]

def select_hard_negatives(triplet):
	#
	negs = []
	for t in triplet[2]:
		#
		negs.append(MODEL.forward(torch.autograd.Variable(t.float().cuda(), volatile=True)))
	negs = torch.cat(negs, 0)
	#
	_, inds = torch.max(torch.mm(MODEL.forward(torch.autograd.Variable(triplet[0].float().cuda())), negs.t()), 1)
	inds = inds.data.long().cpu()
	#
	return [triplet[0], triplet[1], torch.cat(triplet[2], 0).index_select(0, inds)]

#
# loss computation
#

thr = 0.8
beta = -math.log(1.0/0.99 - 1)/(1.0-thr)

def loss_forward(triplet):
	# compute similarities and rescale them to [0, 1]
	AP = torch.mm(triplet[0], triplet[1].t()).add(1).mul(0.5)
	AN = torch.mm(triplet[0], triplet[2].t()).add(1).mul(0.5)
	# kill all scores below `thr`
	AP = F.sigmoid(AP.add(-thr).mul(beta))
	AN = F.sigmoid(AN.add(-thr).mul(beta))
	# compute the loss
	return (1 + torch.sum(torch.max(AN, 1)[0]))/(1 + torch.sum(torch.max(AP, 1)[0]))

def compute_average_loss(triplets):
	# switch to evaluation mode
	MODEL.eval()

	#
	totalloss = 0.0

	for triplet in triplets:
		#
		if isinstance(triplet[2], list):
			triplet = select_hard_negatives(triplet)
		triplet = [
			torch.autograd.Variable(triplet[0].float().cuda(), volatile=True),
			torch.autograd.Variable(triplet[1].float().cuda(), volatile=True),
			torch.autograd.Variable(triplet[2].float().cuda(), volatile=True)
		]

		#
		descs = model_forward(triplet)
		loss = loss_forward(descs)

		totalloss = totalloss + loss.data[0]

	#
	return totalloss/len(triplets)

#
#
#

optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=args.learnrate)
batchsize = args.batchsize

def train_with_sgd(triplets, niters):
	# switch to train mode
	MODEL.train()

	#
	for i in range(0, niters):
		#
		optimizer.zero_grad()
		for j in range(0, batchsize):
			#
			triplet = triplets[ random.randint(0, len(triplets)-1) ]
			if isinstance(triplet[2], list):
				triplet = select_hard_negatives(triplet)
			triplet = [
				torch.autograd.Variable(triplet[0].float().cuda()),
				torch.autograd.Variable(triplet[1].float().cuda()),
				torch.autograd.Variable(triplet[2].float().cuda())
			]
			#
			descs = model_forward(triplet)
			loss = loss_forward(descs)
			loss.div(batchsize)
			loss.backward()
		#
		optimizer.step()

#
# initialize stuff
#

print('* tripletgen: ' + args.tripletgen)
exec(open(args.tripletgen).read())
get_trn_triplets, get_vld_triplets = init()

#
# 
#

t = time.time()
vtriplets = get_vld_triplets()
t = time.time() - t
print("* " + str(len(vtriplets)) + " validation triplets generated in " + str(t) + " [s]")

t = time.time()
ebest = compute_average_loss(vtriplets)
print("* initial validation loss: " + str(ebest))

t = time.time() - t
print("    ** elapsed time: " + str(t) + " [s]")

#
nrounds = 128
for i in range(0, nrounds):
	#
	print("* ROUND (" + str(1+i) + ")")

	#
	t = time.time()
	ttriplets = get_trn_triplets()
	t = time.time() - t
	print("    ** " + str(len(ttriplets)) + " triplets generated in " + str(t) + " [s]")

	#
	t = time.time()
	train_with_sgd(ttriplets, 512)
	t = time.time() - t

	print("    ** elapsed time: " + str(t) + " [s]")

	e = compute_average_loss(ttriplets)
	print("    ** average loss (trn): " + str(e))
	e = compute_average_loss(vtriplets)
	print("    ** average loss (vld): " + str(e))

	if e<ebest and args.writepath:
		#
		print("* saving model parameters to `" + args.writepath + "`")
		MODEL.cpu()
		if args.dataparallel:
			torch.save(MODEL.module.state_dict(), args.writepath)
		else:
			torch.save(MODEL.state_dict(), args.writepath)
		MODEL.cuda()
		#
		ebest = e

	#
	ttriplets = []