import os
import sys
import cv2

dir = sys.argv[2]
os.mkdir(dir)

i = 0

for root, dirnames, filenames in os.walk(sys.argv[1]):
	for filename in filenames:
		if filename == "im0.png":
			#
			l = cv2.imread( os.path.join(root, "im0.png") )
			r = cv2.imread( os.path.join(root, "im1.png") )
			#
			cv2.imwrite(os.path.join(dir, f"{i}-l.jpg"), l)
			cv2.imwrite(os.path.join(dir, f"{i}-r.jpg"), r)
			i = i + 1
