# Learning Local Descriptors from Weakly-Labeled Data

This repo contains the official implementation of the method from the following paper ([arXiv](https://arxiv.org/abs/1603.09095)):

```
@misc
{
	wlrn,
	author = {Nenad Marku\v{s} and Igor S. Pand\v{z}i\'c and J\"{o}rgen Ahlberg},
	title = {{Learning Local Descriptors by Optimizing the Keypoint-Correspondence Criterion}},
	year = {2016},
	eprint = {arXiv:1603.09095}
}
```

This work was presented at the International Conference on Pattern Recognition in December 2016.
Some basic information is available [here](INFO.md).

A newer (updated) version of the paper is available [here](http://hotlab.fer.hr/_download/repository/wlrn.pdf).
We plan to submit it to a journal at some point.

## Requirements

To run the code, you will need:

* PyTorch;
* a CUDA-capable GPU;
* OpenCV for Python (needed for training-data preparation).

## Training

Follow these steps to learn a descriptor for matching ORB keypoints.

#### 1. Prepare the training data

Run `datasets/prepare-ukb-dataset.sh`:

	bash datasets/prepare-ukb-dataset.sh

This script will download the [UKBench dataset](https://archive.org/details/ukbench).

#### 2. Specify the descriptor-extraction structure

The default model-specification class is `models/3x32x32_to_128.py`.

You are encouraged to try different architectures as the default one does not perform very well in all settings.
However, to learn their parameters, some parts of training scripts might need additional tweaking, such as learning rates.

#### 3. Start the learning script

Finally, learn the parameters of the network by running the traininig script:

	python wlrn.py models/3x32x32_to_128.py datasets/tripletgen-ukbench.py --writepath models/3x32x32_to_128.pth

The training should finish in a day or two on a modern GPU
(using the `--dataparallel` flag can reduce training time drastically on a multi-GPU systems).

## Pretrained models

Pretrained models can be downloaded [here](https://drive.google.com/open?id=18ybkdPl-NnAyHLgg0zU-hni__4V7E4XR).

## License

MIT.

## Contact

For any additional information contact me at <nenad.markus@fer.hr>.

Copyright (c) 2016, Nenad Markus. All rights reserved.