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

parser.add_argument('dataloader', type=str, help='a script that loads training and validation samples')
parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')
parser.add_argument('--writepath', type=str, default=None, help='where to write the learned model weights')
parser.add_argument('--learnrate', type=float, default=1e-4, help='RMSprop learning rate')
parser.add_argument('--threshold', type=float, default=0.8, help='WLRN/SKAR threshold')

args = parser.parse_args()

#
#
#

from model import StereoModel
MODEL = StereoModel()

if args.loadpath:
	print('* loading pretrained weights from ' + args.loadpath)
	MODEL.load_state_dict(torch.load(args.loadpath))
MODEL.cuda()

print('* WLRN/SKAR threshold set to %f' % args.threshold)

#
#
#

def compute_matrix_entropy_loss(ammpt, temp=20):
	# ammpt is anchor*positive.t()
	# similarity and probability matrices
	S = ammpt
	P = torch.softmax(temp*S, dim=1)
	cv2.imwrite("P.png", (255*P.detach()).byte().cpu().numpy())
	# compute the average entropy (per row)
	H = - torch.mul(P, torch.log(P))
	H = H.sum() / S.shape[0]
	# we want to minimize entropy (i.e., we want the distribution to be spiky)
	return H

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
	# logging
	#print("  an: %.2f		ap: %.2f" % (torch.sum(torch.max(AN, 1)[0]).item(), torch.sum(torch.max(AP, 1)[0]).item()))
	#print("  an: %.2f		ap: %.2f" % (torch.sum(torch.mean(AN, 1)[0]).item(), torch.sum(torch.mean(AP, 1)[0]).item()))
	cv2.imwrite("an.png", (255*AN.detach()).byte().cpu().numpy())
	cv2.imwrite("ap.png", (255*AP.detach()).byte().cpu().numpy())
	# compute the loss
	H = compute_matrix_entropy_loss( torch.mm(triplet[0], triplet[1].t()) )
	M = (1 + torch.sum(torch.max(AN, 1)[0]))/(1 + torch.sum(torch.max(AP, 1)[0]))
	print(" > M=%.4f, H=%.4f" % (M.detach().item(), H.detach().item()))
	#print("-------------")
	return M + H

# i1/i2 images are 3xHxW tensors
# this loss ranges from 1.0 (very bad, initial values) to 0.0 (not possible to achieve in practice)
# on KITTI, the average loss is around 0.4 or a bit lower when the embedder has been trained
def loss_forward(i1, i2, rowinds, threshold=0.8):
	# compute the features: 1x len(rowinds) xWxD
	i1, i2 = i1.unsqueeze(0), i2.unsqueeze(0)
	f1, f2 = MODEL.forward(i1, i2, rowinds=rowinds)
	f1 = f1[0]
	f2 = f2[0]

	#
	losslist = []
	for r in range(0, f1.shape[0]//3):
		# select anchor, positive and negative sets of embeddings/features
		a = f1[3*r]
		p = f2[3*r]
		n = torch.cat([f1[3*r+1], f1[3*r+2], f2[3*r+1], f2[3*r+2]])
		# accumulate the loss
		losslist.append( compute_triplet_loss((a, p, n), threshold) )

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
		i1 = batch[j][0].cuda()
		i2 = batch[j][1].cuda()
		rowinds = []
		for _ in range(0, 8):
			r = numpy.random.randint(32, i1.shape[1]-8) # randomize row
			rowinds.extend([r, r-3, r+3])
		loss = loss_forward(i1, i2, rowinds, threshold=args.threshold)
		loss.backward()
		avgloss = avgloss + loss.item()
	optimizer.step()
	avgloss = avgloss/len(batch)
	#
	return avgloss

def save_model(name):
	os.system('mkdir -p ' + args.writepath)
	path = args.writepath + '/' + name
	print('* saving model weights to ' + path)
	torch.save(MODEL.state_dict(), path)

for epoch in range(0, 256):
	#
	for i, batch in enumerate(loader):
		start = time.time()
		avgloss = train_step(batch)
		print('* batch %d of epoch %d processed in %.4f [s] (average loss: %f)' % (i, epoch, time.time()-start, avgloss))
		if i%128==0: save_model("ckpt")
	#
	if args.writepath and epoch!=0: save_model(str(epoch) + ".pth")