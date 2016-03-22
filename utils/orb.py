#
import cv2
import sys

#
img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)

n = int(sys.argv[2])

#
orb = cv2.ORB_create(nfeatures=n, patchSize=16)
kpts = orb.detect(img, None)

for kp in kpts:
	#
	print( str(kp.pt[0])+' '+str(kp.pt[1])+' '+str(kp.size)+' '+str(kp.angle))

#
#cv2.imwrite('kptplot.png', cv2.drawKeypoints(img, kpts, None, color=(0, 0, 255), flags=4))