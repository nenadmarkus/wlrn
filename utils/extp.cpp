#include <stdio.h>

// opencv
#include <core/core.hpp>
#include <highgui/highgui.hpp>
#include <features2d/features2d.hpp>
#include <imgproc/imgproc.hpp>

/*
	
*/

namespace PatchExt
{

using namespace cv;

class PatchExtractor: public DescriptorExtractor
{
public:
	PatchExtractor(){side=32; resize=1.5f; usecolor=0; prefix=0;};
	PatchExtractor(int _side, float _resize){side=_side; resize=_resize; usecolor=0; prefix=0;};

	// interface methods inherited from cv::DescriptorExtractor
	virtual void compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const
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
			float* d;
			int r, c;

			//
			rectifyPatch(image, keypoints[i], patchSize, patch);

			if(prefix)
			{
				char buffer[1024];
				sprintf(buffer, "%s%d.png", prefix, i);
				imwrite(buffer, patch);

				//printf("nchannels=%d\n", patch.channels());
				//imshow("...", patch);
				//waitKey(0);
			}

			patch.convertTo(patch, CV_32F);
			patch = patch/255.0f;

			//
			d = descriptors.ptr<float>(i);

			if(!usecolor)
			{
				//
				if(patch.channels()>1)
					cvtColor(patch, patch, CV_BGR2GRAY);

				//
				for(r=0; r<patchSize; ++r)
					for(c=0; c<patchSize; ++c)
						d[r*patchSize+c] = patch.at<float>(r, c);
			}
			else
			{
				//
				int chn;

				//
				for(chn=0; chn<3; ++chn)
					for(r=0; r<patchSize; ++r)
						for(c=0; c<patchSize; ++c)
							d[chn*patchSize*patchSize + r*patchSize + c] = patch.at<Vec3f>(r, c)[chn];
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
	virtual int descriptorType() const {return CV_32FC1;}
	virtual void computeImpl(const Mat&  image, vector<KeyPoint>& keypoints, Mat& descriptors) const
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
	if(argc<2)
	{
		//
		printf("* command line arguments:\n");
		printf("\t** image path (required)\n");
		printf("\t** patch size (in pixels)\n");
		printf("\t** keypoint size multiplier\n");
		printf("\t** output file (if none provided, the programs writes to stdout)\n");

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

	if(argc>=3)
		sscanf(argv[2], "%d", &npix);
	if(argc>=4)
		sscanf(argv[3], "%f", &size);

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
	FILE* file = stdout;

	if(argc==5)
	{
		//
		file = fopen(argv[4], "wb");

		if(!file)
		{
			printf("* cannot write to '%s'\n", argv[4]);
			return 0;
		}
	}

	fwrite("fPAT", 1, 4, file); // magic
	fwrite(&dim, sizeof(int), 1, file);
	fwrite(&descriptors.rows, sizeof(int), 1, file);

	for(i=0; i<descriptors.rows; ++i)
		fwrite(descriptors.ptr<float>(i), sizeof(float), dim, file);

	//
	return 0;
}
