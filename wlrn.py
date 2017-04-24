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

if len(sys.argv) != 4:
	print("* usage: python wlrn.py <model-definition.py> <triplet-generator.py> <best-model-save-path>")
	sys.exit()

MODELFILE = sys.argv[1]
TRIPLETGENFILE = sys.argv[2]
WRITEPATH = sys.argv[3]

#
# model
#

MODELFILE = sys.argv[1]
exec(open(MODELFILE).read())
MODEL = init()
MODEL.cuda()

def model_forward(triplet):
	#
	return [
		MODEL.forward(triplet[0]),
		MODEL.forward(triplet[1]),
		MODEL.forward(triplet[2])
	]

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
	#
	return (1 + torch.sum(torch.max(AN, 1)[0]))/(1 + torch.sum(torch.max(AP, 1)[0]))

def compute_average_loss(triplets):
	# switch to validation mode
	MODEL.eval()

	#
	avgloss = 0.0

	for i in range(0, len(triplets)):
		#
		triplet = [
			torch.autograd.Variable(triplets[i][0].float().cuda()),
			torch.autograd.Variable(triplets[i][1].float().cuda()),
			torch.autograd.Variable(triplets[i][2].float().cuda())
		]

		#
		descs = model_forward(triplet)
		loss = loss_forward(descs)

		avgloss = avgloss + loss.data[0]

	avgloss = avgloss/len(triplets)

	#
	return avgloss

#
#
#

def train_with_sgd(triplets, niters, batchsize, eta):
	# switch to train mode
	MODEL.train()

	#
	optimizer = torch.optim.RMSprop(MODEL.parameters(), lr=eta)
	for i in range(0, niters):
		#
		optimizer.zero_grad()
		for j in range(0, batchsize):
			#
			triplet = triplets[ random.randint(0, len(triplets)-1) ]
			triplet = [
				torch.autograd.Variable(triplet[0]).float().cuda(),
				torch.autograd.Variable(triplet[1]).float().cuda(),
				torch.autograd.Variable(triplet[2]).float().cuda()
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

exec(open(TRIPLETGENFILE).read())
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
elast = ebest
print("* initial validation loss: " + str(ebest))

t = time.time() - t
print("    ** elapsed time: " + str(t) + " [s]")

#
eta = 1e-4
batchsize = 16
nrounds = 128

for i in range(0, nrounds):
	#
	print("* ROUND (" + str(1+i) + ")")

	#
	t = time.time()
	ttriplets = get_trn_triplets()
	t = time.time() - t
	print("    ** " + str(len(ttriplets)) + " triplets generated in " + str(t) + " [s]")

	print("    ** eta=" + str(eta) + ", batch size=" + str(batchsize))

	#
	t = time.time()
	train_with_sgd(ttriplets, 512, batchsize, eta)
	t = time.time() - t

	print("    ** elapsed time: " + str(t) + " [s]")

	e = compute_average_loss(ttriplets)
	print("    ** average loss (trn): " + str(e))
	e = compute_average_loss(vtriplets)
	print("    ** average loss (vld): " + str(e))

	if e<ebest:
		#
		print("* saving the model to `" + WRITEPATH + "`")
		torch.save(MODEL, WRITEPATH)
		#
		ebest = e
	elif elast<e:
		if 64==batchsize:
			#
			eta = eta/2.0
			batchsize = 16
		else:
			#
			batchsize = 2*batchsize

	#
	elast = e

	#
	ttriplets = []