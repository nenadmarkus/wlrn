Old code and idea from 2017!!!

How to reproduce the results:

```
python train.py models/mccnn.py loaders/kitti-stereo2015.py --writepath ./savedir --learnrate 1e-5
python eval.py models.mccnn --loadpath savedir/255.pth
```

Pseudolabels:

```
---
```

Error visualization (KITTI):

```
python eval.py models.mccnn_large --loadpath models/mccnn_large-e255.pth --filtering softmax,10,0.99 --consistency --vizdir VIZ
```