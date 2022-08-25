# UW Madison GI Tract Segmentation Competition

This repo contains my workspace for the UW Madison GI Tract Segmentation Compeition, where I ranked the 23rd and won a silver medal.

My teammates Kolya Forrat and Artem Toporov and our final solution was a blend of 2.5D models and 3D models.

You'll find my approach only in this repo, which is an ensemble of 5 custom Unet++ with deep supervision loss enabled trained on 2.5D data. It's present in `best_config_train.ipynb` notebook.

All of my experiments are present in `experiments.ipynb` notebook, from which you can get a look into how my approach got refined during the competition to achieve my final results.

All of the code that I have written was packaged into the `uwmadisonutils` directory to enable using it across all notebooks during experimentation and development. It contains abstracts of every step including:
1. Loading data
2. Data preprocessing
3. Data augmentations
4. Data loaders preparations
5. Custom models (simple unet++, unet++ with pretrained backbones and simple unet#)
7. Loss functions and metrics
6. Standardized fastai learner creation for different model architectures

The following part is a summary of me and my teamates final approaches in the competition.

## 2.5D approach

5 models trained on 5 folds of the training set split using group stratified split using mixed precision training.

- architecture: Custom made Unet++ to enable deep supervision loss
- backbone: regnety_160 with imagenet weights
- data: (-2, 0,+2) 2.5D Slices - (192, 192) random crops of (320, 384) padded slices with MixUp
- augmentations: HorizontalFlips, ElasticTransform, GridDistortion, ShiftScaleRotate, CoarseDropout, RandomBrightnessContrast, RandomGamma, Sharpen
- loss function: Deep supervision loss of 2*Dice + 3*Focal + BCE for the upper four layers in Unet++ architecutre
- epochs: 100
- optimizer: Adam
- learning rate: 6e-3 with one cycle policy
- inference: tta (horizontal flips only) and minimal post processing removing any predictions from slices that didn’t contain any annotations

I didn’t focus much on post processing strategies or training sampling strategies, and in hind sight, it looks like I should have. But anyways my pipeline was as simple as this, and a single model achieved 0.883 on public LB and 5 folds achieved 0.884.

### Caveats

1. Training with larger crops didn’t improve performance and only increased training time
2. Deep supervision loss was detrimental in the pipeline and significantly improved both local CV and public LB score (0.865 → 0.878)
3. MixUp enabled breaking a certain barrier in local CV and public LB metric score (0.878 → 0.883)
4. RandomResizedCrops didn’t work well as just RandomCrops
5. The final pipeline took around 12 hours to train on one fold using RTX 5000.

## 3d pipeline
We used MONAI Unet multilabel. 3 folds + full train
- channels=(56, 112, 224, 448, 896)
- loss function DiceLoss
- optimizer AdamW
- lr_mode warmup_restart every 100 epoch
Train it on raw data with random crops 160, 160, 80

Inference on whole case_day for length =80 and for length=144 we crop it to 124 and 3 separate inference with step =20 with sliding window, roi_size=(288,288,80) , overlap=0.9

Augmentations
RandFlipd, RandAffined, RandGridDistortiond, RandScaleIntensityd, RandShiftIntensity

### Didn't work for 3d:
- deep supervision loss
- mixup
- data cleansing
- pseudo labeling on samples with low metric
- train on only areas where is masks ob train for each size


As postprocessing we used few technics:
- Delete impossible - it is deleting all slices on which we don't have any masks in train set
- Fill missing - filling slices by nearest if s-1 and s+1 is no empty
- Clustering - in train set every mask appears on some slice and expire on some other. We found biggest cluster for every case_day and delete everything what not in cluster
- Delete last N - deleting last n, m masks for case_days with Z-dim = 144 and 80. Best values m = n = 2

It added + ~0.004 to LB


## 2.5D and 3D Ensemble

Ensembling of 2.5d * 0.7 + 3d * 0.3 added ~0.004
