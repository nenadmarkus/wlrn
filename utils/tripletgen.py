#
import torch
import os
import random

import torchvision.transforms as transforms

from PIL import Image

#
def generate_triplets(bags):
	#
	triplets = []

	for i in range(0, len(bags)):
		for j in range(i+1, len(bags)):
			if bags[i][1] == bags[j][1]: # compare labels
				for k in range(0, 3):
					#
					stop = False
					while not stop:
						q = random.randint(0, len(bags)-1)
						if bags[i][1] != bags[q][1]:
							stop = True
					#
					triplets.append([
						bags[i][0],
						bags[j][0],
						bags[q][0]
					])

	#
	return triplets

#
def load_keypoint_bags(folder, prob):
	#
	bags = []
	totensor = transforms.ToTensor()
	#
	for root, dirs, files in os.walk(folder):
		for f in files:
			if (f.endswith(".jpg") or f.endswith(".png")) and random.random()<=prob:
				#
				data = totensor( Image.open(os.path.join(root, f)) ).mul(255).byte()
				#
				data = data.view(3, int(data.size(1)/data.size(2)), data.size(2), data.size(2)).transpose(0, 1).contiguous()
				#
				label = f.split('.')[0]
				#
				bags.append( [data, label] )

	#
	return bags

def get_trn_triplets():
	#
	TRN_FOLDER = --TRN-FOLDER--
	TRN_PROB = --TRN-PROBABILITY--
	return generate_triplets(load_keypoint_bags(TRN_FOLDER, TRN_PROB))

def get_vld_triplets():
	#
	VLD_FOLDER = --VLD-FOLDER--
	VLD_PROB = --VLD-PROBABILITY--
	return generate_triplets(load_keypoint_bags(VLD_FOLDER, VLD_PROB))

def init():
	#
	return get_trn_triplets, get_vld_triplets