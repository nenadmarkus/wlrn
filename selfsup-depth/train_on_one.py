import torch
import numpy
import time
import math
import cv2

#
# parse command line options
#

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('modeldef', type=str, help='a script that defines the segmentation network')
parser.add_argument('img_l', type=str, help='left image')
parser.add_argument('img_r', type=str, help='right image')
parser.add_argument('--loadpath', type=str, default=None, help='path from which to load pretrained weights')
parser.add_argument('--writepath', type=str, default=None, help='where to write the learned model weights')
parser.add_argument('--learnrate', type=float, default=1e-5, help='RMSprop learning rate')
parser.add_argument('--threshold', type=float, default=0.8, help='WLRN/SKAR threshold')
parser.add_argument('--usecolor', action='store_true', default=False, help='color images as input or grayscale')
parser.add_argument('--dataparallel', action='store_true', default=False, help='wrap the model into a torch.nn.DataParallel module for multi-gpu learning')

args = parser.parse_args()

#
#
#

l = cv2.imread(args.img_l, cv2.IMREAD_COLOR)[:(11*32),:(38*32),:]
r = cv2.imread(args.img_r, cv2.IMREAD_COLOR)[:(11*32),:(38*32),:]

l = torch.from_numpy(l).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
r = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float().div(255.0)

sample = torch.cat((l, r))

if not args.usecolor:
	# take just the green channel
	sample = sample[:, 1, :, :]
	sample = sample.unsqueeze(1).contiguous()

#
#
#

exec(open(args.modeldef).read())
if args.usecolor:
	MODEL = init(3)
	print("* input channels: 3")
else:
	MODEL = init(1)
	print("* input channels: 1")
if args.loadpath:
	print('* loading pretrained weights from ' + args.loadpath)
	MODEL.load_state_dict(torch.load(args.loadpath, map_location=torch.device('cpu')))

if torch.cuda.is_available(): MODEL.cuda()

if args.dataparallel:
	print('* using nn.DataParallel')
	MODEL = torch.nn.DataParallel(MODEL)

print('* WLRN/SKAR threshold set to %f' % args.threshold)

#
#
#

def compute_matrix_entropy_loss(ammpt, temp=20):
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

def compute_maxprob_loss(ammpt, temp=20):
	# ammpt is anchor*positive.t()
	# similarity and probability matrices
	S = ammpt
	P = torch.softmax(temp*S, dim=1)
	if numpy.random.random()<0.1: cv2.imwrite("P_maxprobloss.png", (255*P.detach()).byte().cpu().numpy())
	# extract just the first part of matrix, where the matches lie
	P = P[:, :P.shape[0]]
	# we want to maximize the max values == maximize the logarithm of max values == minimize negative logarithm max values
	M = - torch.log( torch.max(P, dim=1).values )
	return M.sum() / S.shape[0]

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

def get_match_prob(x, y, temp=20):
	S = torch.mm(x, y.t())
	P0 = torch.softmax(temp*S, dim=0)
	P1 = torch.softmax(temp*S, dim=1)
	P = P0*P1
	return P

# left/right features are DxHxW tensors computed with the embeddings model (e.g., MCCNN) from the left and right stereo image, respectively
# this loss ranges from 1.0 (very bad, initial values) to 0.0 (not possible to achieve in practice)
# on KITTI, the average loss is around 0.4 or a bit lower when the embedder has been trained
def loss_forward(left_features, right_features, threshold=0.8):
	# features dimension as last: DxHxW -> HxWxD
	descs0 = left_features.permute(1, 2, 0)
	descs1 = right_features.permute(1, 2, 0)

	# iterate over 16 lines for this stereo pair (16 is some constant that works well)
	losslist = []
	for i in range(0, 128):
		# select the image/featres row (in H dimension)
		r = numpy.random.randint(16, descs0.shape[0]-16)
		#'''
		# select anchor, positive and negative sets of embeddings/features
		a = descs0[r]
		p = descs1[r]
		n = torch.cat([descs0[r-3], descs0[r+3], descs1[r-3], descs1[r+3]])
		# accumulate the loss
		losslist.append( compute_triplet_loss((a, p, n), threshold) )
		#'''

		'''
		a = descs0[r]
		o = torch.cat([descs1[r], descs0[r-3], descs0[r+3], descs1[r-3], descs1[r+3]])
		M = torch.mm(a, o.t())
		losslist.append( compute_maxprob_loss(M) )
		'''

		a = descs0[r]
		p = descs1[r]
		P = get_match_prob(a, p, temp=20)
		if i==15:
			cv2.imwrite("P.png", (255*P.detach()).byte().cpu().numpy())
			#cv2.imshow("matchprob", (255*P.detach()).byte().cpu().numpy())
			#cv2.waitKey(1)
			#print(torch.max(P, axis=0)[0].detach().tolist())

		#H = - torch.mul(P, torch.log(P))
		#H = H.sum() / P.shape[0]
		#losslist.append(H)

	# we're done: average the loss
	return sum(losslist)/len(losslist)

#
#
#



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
		featuremaps = MODEL(batch[j].cuda() if torch.cuda.is_available() else batch[j])
		featuremaps = torch.nn.functional.normalize(featuremaps, p=2, dim=1) # L2 normalize
		loss = loss_forward(featuremaps[0], featuremaps[1], threshold=args.threshold)
		loss.backward()
		avgloss = avgloss + loss.item()
	optimizer.step()
	avgloss = avgloss/len(batch)
	#
	return avgloss

for iter in range(0, 256):
	#
	start = time.time()
	avgloss = train_step([sample])
	print('* iter %d processed in %.4f [s] (average loss: %f)' % (iter, time.time()-start, avgloss))
	#
	'''
	if args.writepath and iter!=0:
		os.system('mkdir -p ' + args.writepath)
		path = args.writepath + '/' + str(epoch) + '.pth'
		print('* saving model weights to ' + path)
		if args.dataparallel:
			torch.save(MODEL.module.state_dict(), path)
		else:
			torch.save(MODEL.state_dict(), path)
	'''