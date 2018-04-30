#
import torch
import os
import numpy
import cv2

#
cv2.setNumThreads(0)

#
def generate_triplets(bags):
	#
	triplets = []

	for i in range(0, len(bags)):
		for j in range(i+1, len(bags)):
			if bags[i][1] == bags[j][1]: # compare labels
				#
				negbags = []
				#
				for k in range(0, 12):
					#
					stop = False
					while not stop:
						q = numpy.random.randint(0, len(bags))
						if bags[i][1] != bags[q][1]:
							stop = True
					#
					negbags.append(bags[q][0])
				#
				usehardnegs = True
				if usehardnegs:
					triplets.append([
						bags[i][0],
						bags[j][0],
						negbags
					])
				else:
					for negbag in negbags:
						triplets.append([
							bags[i][0],
							bags[j][0],
							negbag
						])

	#
	return triplets

#
def extract_patches(img, keypoints, npix, size):
	#
	patches = []
	for x, y, s, a in keypoints:
		#
		s = size*s/npix
		cos = numpy.cos(a*numpy.pi/180.0)
		sin = numpy.sin(a*numpy.pi/180.0)
		#
		M = numpy.matrix([
			[+s*cos, -s*sin, (-s*cos+s*sin)*npix/2.0 + x],
			[+s*sin, +s*cos, (-s*sin-s*cos)*npix/2.0 + y]
		])
		#
		p = cv2.warpAffine(img, M, (npix, npix), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_CUBIC+cv2.WARP_FILL_OUTLIERS)
		patches.append( torch.from_numpy(p).permute(2, 0, 1) )
	#
	if len(patches) < 16:
		return None
	else:
		return torch.stack(patches)

#
def load_keypoint_bags(imgpaths, prob):
	#
	orb = cv2.ORB_create(nfeatures=512)
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=512)
	def get_keypoints(img):
		keypoints = []
		keypoints.extend(orb.detect(img, None))
		keypoints.extend(sift.detect(img, None))
		return [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in keypoints]
	#
	bags = []
	for imgpath in imgpaths:
		if numpy.random.random()<=prob:
			#
			#print('* processing ' + imgpath.split('/')[-1])
			img = cv2.imread(imgpath)
			keypoints = get_keypoints(img)
			patches = extract_patches(img, keypoints, 32, 1.5)
			#
			label = int(imgpath.split('/')[-1][7:12])//4
			#
			if patches is not None:
				bags.append( [patches, label] )
	#
	return bags

def init(folder='datasets/ukbench'):
	#
	trn = []
	vld = []
	for root, dirnames, filenames in os.walk(folder):
		for filename in filenames:
			if filename.endswith('.jpg'):
				if int(filename[7:12])//4<300:
					vld.append(os.path.join(root, filename))
				else:
					trn.append(os.path.join(root, filename))
	#
	return (lambda: generate_triplets(load_keypoint_bags(trn, 0.5))), (lambda: generate_triplets(load_keypoint_bags(vld, 1.00)))

#
'''
a, b = init('ukbench')
import time
start = time.time()
trn = a()
print('* elapsed time: %f [s]' % (time.time()-start))
print(len(trn))
'''