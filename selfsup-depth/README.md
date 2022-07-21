Old code and idea from 2017!!!

How to reproduce the results:

```
python train.py models/mccnn.py loaders/kitti-stereo2015.py --writepath ./savedir --learnrate 1e-5
python eval.py models.mccnn --loadpath savedir/255.pth
```
