import os
import sys
import cv2

dir = sys.argv[1]
os.mkdir(dir)

exec(open("loaders/kitti-stereo2015.py").read())
load_sample, n = get_load_sample("/mnt/ssd1/data/kitti2015/data_scene_flow_multiview/training/")
os.mkdir(os.path.join(dir, "kitti2015"))
for i in range(0, n):
	l, r = load_sample(i)
	cv2.imwrite(os.path.join(dir, "kitti2015", f"{i}-l.jpg"), l)
	cv2.imwrite(os.path.join(dir, "kitti2015", f"{i}-r.jpg"), r)
	#cv2.imshow("...", l)
	#cv2.waitKey(0)

exec(open("loaders/kitti-stereo2012.py").read())
load_sample, n = get_load_sample("/mnt/ssd1/data/kitti2012/data_stereo_flow_multiview/training/")
os.mkdir(os.path.join(dir, "kitti2012"))
for i in range(0, n):
	l, r = load_sample(i)
	cv2.imwrite(os.path.join(dir, "kitti2012", f"{i}-l.jpg"), l)
	cv2.imwrite(os.path.join(dir, "kitti2012", f"{i}-r.jpg"), r)
	#cv2.imshow("...", l)
	#cv2.waitKey(0)
