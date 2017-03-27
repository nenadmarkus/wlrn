# Learning Local Descriptors from Weakly-Labeled Data

Current best local descriptors are learned on a large dataset of matching and non-matching keypoint pairs.
However, data of this kind is not always available since detailed keypoint correspondences can be hard to establish (e.g., for non-image data).
On the other hand, we can often obtain labels for pairs of keypoint bags.
For example, keypoint bags extracted from two images of the same object under different views form a matching pair, and keypoint bags extracted from images of different objects form a non-matching pair.
On average, matching pairs should contain more corresponding keypoints than non-matching pairs.
We propose to learn local descriptors from such information where local correspondences are not known in advance.

<center><img src="teaser.png" alt="Teaser" style="width: 512px;"/></center>

Each image in the dataset (first row) is processed with a keypoint detector (second row) and transformed into a bag of visual words (third row).
Some bags form matching pairs (green arrow) and some form non-matching pairs (red arrows).
On average, matching pairs should contain more corresponding local visual words than non-matching pairs.
We propose to *learn local descriptors* by optimizing the mentioned local correspondence criterion on a given dataset.
Note that prior work assumes local correspondences are known in advance.