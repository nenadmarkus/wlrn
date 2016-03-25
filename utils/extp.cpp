#include <stdio.h>
#include <vector>

// opencv 3.x required
#include <opencv2/opencv.hpp>

/*
	
*/

#define USECOLOR 1

namespace PatchExt
{

using namespace cv;

class PatchExtractor: public DescriptorExtractor
{
public:
	PatchExtractor(){side=32; resize=1.5f; usecolor=USECOLOR; prefix=0;};
	PatchExtractor(int _side, float _resize){side=_side; resize=_resize; usecolor=USECOLOR; prefix=0;};

	// interface methods inherited from cv::DescriptorExtractor
	virtual void compute(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) const
	{
		int i, patchSize;

		//
		if(usecolor)
			descriptors = Mat::zeros(keypoints.size(), 3*side*side, descriptorType());
		else
			descriptors = Mat::zeros(keypoints.size(), side*side, descriptorType());

		//
		patchSize = side;

		for(i=0; i<keypoints.size(); ++i)
		{
			Mat patch;
			uint8_t* d;
			int r, c;

			//
			rectifyPatch(image, keypoints[i], patchSize, patch);
			patch.convertTo(patch, CV_8U);

			//printf("nchannels=%d\n", patch.channels());
			//imshow("...", patch);
			//waitKey(0);

			if(prefix)
			{
				char buffer[1024];
				sprintf(buffer, "%s%d.png", prefix, i);
				imwrite(buffer, patch);
			}

			//
			d = descriptors.ptr<uint8_t>(i);

			if(!usecolor)
			{
				//
				if(patch.channels()>1)
					cvtColor(patch, patch, CV_BGR2GRAY);

				//
				for(r=0; r<patchSize; ++r)
					for(c=0; c<patchSize; ++c)
						d[r*patchSize+c] = patch.at<uint8_t>(r, c);
			}
			else
			{
				//
				int chn;

				//
				for(chn=0; chn<3; ++chn)
					for(r=0; r<patchSize; ++r)
						for(c=0; c<patchSize; ++c)
							d[chn*patchSize*patchSize + r*patchSize + c] = patch.at<Vec3b>(r, c)[chn];
			}

			//
			patch.release();
		}
	}

	virtual int descriptorSize() const
	{
		if(usecolor)
			return 3*side*side;
		else
			return side*side;
	}
	virtual int descriptorType() const {return CV_8U;}
	virtual void computeImpl(const Mat&  image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		compute(image, keypoints, descriptors);
	}

	//
	char* prefix;

private:
	int side, usecolor;
	float resize;

	void rectifyPatch(const Mat& image, const KeyPoint& kp, const int& patchSize, Mat& patch) const
	{
		float s = resize * (float) kp.size / (float) patchSize;

		float cosine = (kp.angle>=0) ? cos(kp.angle*CV_PI/180) : 1.f;
		float sine   = (kp.angle>=0) ? sin(kp.angle*CV_PI/180) : 0.f;

		float M_[] =
		{
			s*cosine, -s*sine,   (-s*cosine + s*sine  ) * patchSize/2.0f + kp.pt.x,
			s*sine,   s*cosine,  (-s*sine   - s*cosine) * patchSize/2.0f + kp.pt.y
		};

		//
		warpAffine(image, patch, Mat(2, 3, CV_32FC1, M_), Size(patchSize, patchSize), CV_WARP_INVERSE_MAP + CV_INTER_CUBIC + CV_WARP_FILL_OUTLIERS);
	}
};

}

/*
	
*/

int main(int argc, char* argv[])
{
	int i, dim, type;
	char* magic;

	//
	if(argc<4)
	{
		//
		printf("* command line arguments:\n");
		printf("\t** image path\n");
		printf("\t** patch size (in pixels)\n");
		printf("\t** keypoint size multiplier\n");
		printf("\t** optional: output file (if none provided, the programs writes to stdout)\n");

		//
		return 0;
	}

	//
	cv::Mat im = cv::imread(argv[1]);

	if(!im.data)
		return 0;

	//
	cv::Mat gray = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	//
	std::vector<cv::KeyPoint> keypoints;

	float x, y, s, a;

	while(4 == scanf("%f %f %f %f", &x, &y, &s, &a))
	{
		//
		cv::KeyPoint keypoint;

		keypoint.pt.x = x;
		keypoint.pt.y = y;
		keypoint.size = s;
		keypoint.angle = a;

		//
		keypoints.push_back(keypoint);
	}

	//
	int npix=32;
	float size=1.5f;

	sscanf(argv[2], "%d", &npix);
	sscanf(argv[3], "%f", &size);

	//
	cv::Mat descriptors;
	PatchExt::PatchExtractor patchExtractor(npix, size);

	// uncomment if you'd like to see how the extracted patches look like
	/*
	system("mkdir -p patches");
	patchExtractor.prefix = (char*)"patches/p";
	//*/

	patchExtractor.compute(im, keypoints, descriptors);
	dim = patchExtractor.descriptorSize();

	//
	if(argc==5)
	{
		if(!USECOLOR)
		{
			int i, r, c;

			cv::Mat img(descriptors.rows*npix, npix, CV_8UC1);

			for(i=0; i<descriptors.rows; ++i)
				for(r=0; r<npix; ++r)
					for(c=0; c<npix; ++c)
						img.at<uint8_t>(i*npix+r, c) = descriptors.at<uint8_t>(i, r*npix + c);

			cv::imwrite(argv[4], img);
		}
		else
		{
			int i, r, c;

			cv::Mat img(descriptors.rows*npix, npix, CV_8UC3);

			for(i=0; i<descriptors.rows; ++i)
			{
				//
				uint8_t* d = descriptors.ptr<uint8_t>(i);

				//
				for(r=0; r<npix; ++r)
					for(c=0; c<npix; ++c)
					{
						img.at<cv::Vec3b>(i*npix+r, c)[0] = d[0*npix*npix + r*npix + c];
						img.at<cv::Vec3b>(i*npix+r, c)[1] = d[1*npix*npix + r*npix + c];
						img.at<cv::Vec3b>(i*npix+r, c)[2] = d[2*npix*npix + r*npix + c];
					}
			}

			//

			std::vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);

			cv::imwrite(argv[4], img, compression_params);
		}
	}
	else
	{
		//
		FILE* file = stdout;

		fwrite("Bpch", 1, 4, file); // magic
		fwrite(&dim, sizeof(int), 1, file);
		fwrite(&descriptors.rows, sizeof(int), 1, file);

		for(i=0; i<descriptors.rows; ++i)
			fwrite(descriptors.ptr<uint8_t>(i), sizeof(uint8_t), dim, file);
	}

	//
	return 0;
}
