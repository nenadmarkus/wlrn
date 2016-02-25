#include <stdio.h>
#include <math.h>

// opencv
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <features2d/features2d.hpp>
#include <imgproc/imgproc.hpp>

/*
	
*/

float estrot(int r, int c, uint8_t pixels[], int nrows, int ncols, int ldim)
{
	int mr, mc;

	const int radius = 15;

	//
	if(r-radius<=0 || r+radius>=nrows-1 || c-radius<=0 || c+radius>=ncols-1)
		return 0.0f;

	//
	pixels = &pixels[r*ldim+c];

	//
	mr = mc = 0;

	for(r=-radius; r<=+radius; ++r)
		for(c=-radius; c<=+radius; ++c)
		{
			mr = mr + r*pixels[r*ldim+c];
			mc = mc + c*pixels[r*ldim+c];
		}

	//
	return atan2(mr, mc);
}

void extract_keypoints(cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, int n)
{
	int i, t;

	//
	if(img.channels() != 1)
		return;

	//
	t = 200;

	while( keypoints.size()<n && t>0 )
	{
		cv::FAST(img, keypoints, t);
		t = t - 2;
	}

	//
	for(i=0; i<keypoints.size(); ++i)
	{
		keypoints[i].angle = MAX(0, 180.0f/CV_PI*(CV_PI + estrot(keypoints[i].pt.y, keypoints[i].pt.x, img.data, img.rows, img.cols, img.step)));
		keypoints[i].size = 12;
	}
}

/*
void draw_keypoints(const char* src, const char* dst, int n)
{
	int i;

	//
	cv::Mat gray = cv::imread(src, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat rgb= cv::imread(src);

	//
	std::vector<cv::KeyPoint> keypoints;

	cv::resize(gray, gray, cv::Size(gray.cols/2, gray.rows/2));
	extract_keypoints(gray, keypoints, n);
	for(i=0; i<keypoints.size(); ++i)
	{
		keypoints[i].pt.x*=2;
		keypoints[i].pt.y*=2;
	}

	//
	cv::Mat descriptors;
	PatchExt::PatchExtractor patchExtractor(64, 8.0f);
	patchExtractor.prefix = (char*)"tmp/patches/";
	patchExtractor.compute(rgb, keypoints, descriptors);

	//
	for(i=0; i<keypoints.size(); ++i)
	{
		float r, c, s, t;

		//
		c = cos(keypoints[i].angle*CV_PI/180.0f);
		s = sin(keypoints[i].angle*CV_PI/180.0f);

		//
		r = rgb.rows/32;

		t = -1;

		cv::circle(rgb, cv::Point(keypoints[i].pt.x, keypoints[i].pt.y), r, CV_RGB(0, 255, 0), t);
		//cv::line(rgb, cv::Point(keypoints[i].pt.x, keypoints[i].pt.y), cv::Point(keypoints[i].pt.x+r*c, keypoints[i].pt.y+r*s), CV_RGB(0, 255, 0), t);
	}

	//
	cv::imwrite("tmp/out.jpg", rgb);
}
*/

/*
	
*/

int main(int argc, char* argv[])
{
	int i, n;

	//
	if(argc<3)
	{
		//
		printf("* command line arguments:\n");
		printf("\t** image\n");
		printf("\t** num keypoints to extract\n");

		//
		return 0;
	}

	//
	cv::Mat im = cv::imread(argv[1]);

	if(!im.data)
		return 0;

	cv::Mat gray;
	cv::cvtColor(im, gray, CV_RGB2GRAY);

	//
	sscanf(argv[2], "%d", &n);

	std::vector<cv::KeyPoint> keypoints;
	extract_keypoints(gray, keypoints, n);

	//
	for(i=0; i<keypoints.size(); ++i)
		printf("%f %f %f %f\n", keypoints[i].pt.x, keypoints[i].pt.y, keypoints[i].size, keypoints[i].angle);

	//
	return 0;
}
