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

* Torch7 or PyTorch (both are supported);
* a CUDA-capable GPU, cuDNN;
* Python with OpenCV (needed for training-data preparation).

## Training

Follow these steps.

#### 1. Prepare the training data

Run `datasets/prepare-ukb-dataset.sh`:

	bash datasets/prepare-ukb-dataset.sh

The script will download the UKB dataset, extract keypoints from the images and store them in an appropriate format.
The script will also prepare data-loading routines by modifying the `utils/tripletgen.lua` and `utils/tripletgen.py` templates.

#### 2. Specify the descriptor-extraction structure

For Troch7-based implementation, you can generate the default model by running `th models/3x32x32_to_64.lua models/net.t7`.
For PyTorch, the default model-specification class is `models/3x32x32_to_128.py`.

You are encouraged to try different architectures as the default one does not perform very well in all settings.
However, to learn their parameters, some parts of training scripts might need additional tweaking, such as learning rates.

#### 3. Start the learning script

Finally, learn the parameters of the network by running the traininig script.

For Troch7, run

	th wlrn.lua models/net.t7 datasets/tripletgen-ukb.lua -w models/net-trained.t7

For PyTorch, run

	python wlrn.py models/3x32x32_to_128.py datasets/tripletgen-ukb.py models/net.pytorch

The training should finish in a couple of hours on a modern GPU.

## Pretrained models

You can download some pretrained models from <https://nenadmarkus.com/data/wlrn-pretrained.zip>.

## License

MIT.

## Contact

For any additional information contact me at <nenad.markus@fer.hr>.

Copyright (c) 2016, Nenad Markus. All rights reserved.