Old code and idea from 2017!!!

How to reproduce the results:

```
python train.py models.mccnn_large loaders/kitti-stereo2015.py --writepath ./savedir --learnrate 1e-5
python eval.py models.mccnn_large --loadpath savedir/255.pth
```

Error visualization (KITTI):

```
python eval.py models.mccnn_large --loadpath models/mccnn_large.params --filtering softmax,10,0.99 --consistency --vizdir VIZ
```

Generating labels for training a deep stereo model:

```
python make_pseudo_labels.py models.mccnn_large models/mccnn_large.params softmax,10,0.99 datasets/LRD
```