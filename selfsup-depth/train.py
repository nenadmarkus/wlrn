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
parser.add_argument('--threshold', type=float, default=0.8, help='WLRN/SKAR threshold')
parser.add_argument('--usecolor', action='store_true', default=True, help='color images as input or grayscale')
parser.add_argument('--dataparallel', action='store_true', default=False, help='wrap the model into a torch.nn.DataParallel module for multi-gpu learning')

args = parser.parse_args()

#
#
#

exec(open(args.modeldef).read())
if args.usecolor: MODEL = init(3)
else: MODEL = init(1)
if args.loadpath:
	print('* loading pretrained weights from ' + args.loadpath)
	MODEL.load_state_dict(torch.load(args.loadpath))
MODEL.cuda()

if args.dataparallel:
	print('* using nn.DataParallel')
	MODEL = torch.nn.DataParallel(MODEL)

print('* WLRN/SKAR threshold set to %f' % args.threshold)

#
#
#

def compute_matrix_entropy_loss(ammpt, temp=30):
	# ammpt is anchor*positive.t()
	# similarity and probability matrices
	S = ammpt
	P = torch.softmax(temp*S, dim=1)
	if numpy.random.random()<0.1: cv2.imwrite("P.png", (255*P.detach()).byte().cpu().numpy())
	# compute the average entropy (per row)
	H = - torch.mul(P, torch.log(P))
	H = H.sum() / S.shape[0]
	# we want to minimize entropy (i.e., we want the distribution to be spiky)
	return H

def _loss(triplet, thr):
	beta = -math.log(1.0/0.99 - 1)/(1.0-thr)
	AP = torch.mm(triplet[0], triplet[1].t()).add(1).mul(0.5)
	AP = torch.sigmoid(AP.add(-thr).mul(beta))
	return ( compute_matrix_entropy_loss(AP) + compute_matrix_entropy_loss(AP.t()) )/2.0

# auxiliary function
# computes the loss for the (anchor, positive, negative) bags of embeddings
# see <https://arxiv.org/abs/1603.09095> for an explanation
def compute_triplet_loss(triplet, thr):
	# this is a parameter of the loss
	beta = -math.log(1.0/0.99 - 1)/(1.0-thr)
	# compute similarities and rescale them to [0, 1]
	AP = torch.mm(triplet[0], triplet[1].t()).add(1).mul(0.5)
	AN = torch.mm(triplet[0], triplet[2].t()).add(1).mul(0.5)
	# kill all scores below `thr`
	AP = torch.sigmoid(AP.add(-thr).mul(beta))
	AN = torch.sigmoid(AN.add(-thr).mul(beta))
	# compute the loss
	M = (1 + torch.sum(torch.max(AN, 1)[0]))/(1 + torch.sum(torch.max(AP, 1)[0]))
	#cv2.imwrite("an_.png", (255*AN.detach()).byte().cpu().numpy())
	#cv2.imwrite("ap_.png", (255*AP.detach()).byte().cpu().numpy())
	return M

# left/right features are DxHxW tensors computed with the embeddings model (e.g., MCCNN) from the left and right stereo image, respectively
# this loss ranges from 1.0 (very bad, initial values) to 0.0 (not possible to achieve in practice)
# on KITTI, the average loss is around 0.4 or a bit lower when the embedder has been trained
def loss_forward(left_features, right_features, threshold=0.8):
	# features dimension as last: DxHxW -> HxWxD
	descs0 = left_features.permute(1, 2, 0)
	descs1 = right_features.permute(1, 2, 0)

	# iterate over 16 lines for this stereo pair (16 is some constant that works well)
	losslist = []
	for i in range(0, 16):
		# select the image/featres row (in H dimension)
		r = numpy.random.randint(16, descs0.shape[0]-16)
		# select anchor, positive and negative sets of embeddings/features
		a = descs0[r]
		p = descs1[r]
		n = torch.cat([descs0[r-3], descs0[r+3], descs1[r-3], descs1[r+3]])
		# accumulate the loss
		losslist.append( compute_triplet_loss((a, p, n), threshold) )
		#losslist.append( (compute_matrix_entropy_loss(torch.mm(a, p.t()))+compute_matrix_entropy_loss(torch.mm(p, a.t())))/2.0 )
		#losslist.append( _loss((a, p, n), threshold) )

	# we're done: average the loss
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
		featuremaps = torch.nn.functional.normalize(featuremaps, p=2, dim=1) # L2 normalize
		loss = loss_forward(featuremaps[0], featuremaps[1], threshold=args.threshold)
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
