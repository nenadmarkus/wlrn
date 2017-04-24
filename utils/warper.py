#
import cv2
import numpy
import sys
import math

#
imgpath = sys.argv[1]
npix = int(sys.argv[2])
size = float(sys.argv[3])
outpath = sys.argv[4]

#
img = cv2.imread(imgpath)

#
lines = sys.stdin.readlines()
patches = []
for line in lines:
	#
	x = float(line.split()[0])
	y = float(line.split()[1])
	s = float(line.split()[2])
	a = float(line.split()[3])
	#
	s = size*s/npix
	cos = math.cos(a*math.pi/180.0)
	sin = math.sin(a*math.pi/180.0)
	#
	M = numpy.matrix([
		[+s*cos, -s*sin, (-s*cos+s*sin)*npix/2.0 + x],
		[+s*sin, +s*cos, (-s*sin-s*cos)*npix/2.0 + y]
	])
	#
	p = cv2.warpAffine(img, M, (npix, npix), flags=cv2.WARP_INVERSE_MAP+cv2.INTER_CUBIC+cv2.WARP_FILL_OUTLIERS)
	patches.append(p)

patches = numpy.vstack(patches)

#
cv2.imwrite(outpath, patches)