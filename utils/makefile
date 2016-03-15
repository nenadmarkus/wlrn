#
#
#

OPENCV = -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -I/usr/include/opencv2

#
#
#

build:
	c++ extp.cpp $(OPENCV) -o extp
	c++ fast.cpp -lm $(OPENCV) -o fast
	cc nn.c -lm -fopenmp -o nn

