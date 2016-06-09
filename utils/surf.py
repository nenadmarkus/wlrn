#
import cv2
import sys

#
img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)

n = int(sys.argv[2])

#
hesshi = 32768
hesslo = 0

stop = False

while not stop:
	#
	hess = (hesshi + hesslo)/2.0

	#
	surf = cv2.xfeatures2d.SURF_create(hess, 4, 2, True, False)
	kpts = surf.detect(img, None)

	#
	if abs(len(kpts)-n)<16 or abs(hesshi-hesslo)<32:
		stop = True
	elif len(kpts)<n:
		hesshi = hess
	else:
		hesslo = hess

	hess = hess/2

for kp in kpts:
	#
	print( str(kp.pt[0])+' '+str(kp.pt[1])+' '+str(kp.size)+' '+str(kp.angle))

#
#cv2.imwrite('kptplot.png', cv2.drawKeypoints(img, kpts, None, color=(0, 0, 255), flags=4))