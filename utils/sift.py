#
import cv2
import sys

#
img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2GRAY)

n = int(sys.argv[2])

#
#sift = cv2.xfeatures2d.SIFT_create(nfeatures=n, nOctaveLayers=128)
sift = cv2.xfeatures2d.SIFT_create(nfeatures=n)
kpts = sift.detect(img, None)

for kp in kpts:
	#
	if kp.size < 12:
		kp.size = 12

	print( str(kp.pt[0])+' '+str(kp.pt[1])+' '+str(kp.size)+' '+str(kp.angle))

#
#cv2.imwrite('kptplot.png', cv2.drawKeypoints(img, kpts, None, color=(0, 0, 255), flags=4))

# DUPLICATE KEYPOINTS EXPLAINED:
# 	http://stackoverflow.com/questions/10828501/duplicate-sift-keypoints-in-a-single-image