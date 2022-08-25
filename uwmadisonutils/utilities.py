import os
import gc
import matplotlib.patches as mpatches
import albumentations

import segmentation_models_pytorch as smp

from tqdm import tqdm

from itertools import chain

from sklearn.model_selection import GroupShuffleSplit
from fastai.vision.all import *

## Helper Functions
# Extract case id from fname
def get_case_id(fname):
    if KAGGLE: i = 5
    elif GRADIENT: i = 2
    return fname.parts[i] + '_' + fname.parts[i+2][:10]

def check_file(file_id, fname):
    case_id, day, _, slice_no = file_id.split('_')
    if case_id == fname.parts[1] and day == fname.parts[2].split('_')[1] and slice_no in fname.parts[-1]:
        return True
    return False

def get_file(file_id):
    return fnames.filter(lambda f: check_file(not_null_train.id[0], f))[0]

# https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-eda
def get_custom_df(df, fnames, root, channels=3, stride=1):    
    # 1. Get Case-ID as a column (str and int)
    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(lambda x: int(x.split("_", 2)[0].replace("case", "")))

    # 2. Get Day as a column
    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
    df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))
    
    # Add case_day str column
    df["case_id_day_num_str"] = df["case_id_str"] + "_" + df["day_num_str"]

    # 3. Get Slice Identifier as a column
    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    # 4. Get full file paths for the representative scans
    df["_partial_fname"] = (root+'/'+ # /kaggle/input/uw-madison-gi-tract-image-segmentation/train/
                          df["case_id_str"]+"/"+ # .../case###/
                          df["case_id_str"]+"_"+df["day_num_str"]+ # .../case###_day##/
                          "/scans/"+df["slice_id"]) # .../slice_####
    
    _tmp_merge_df = pd.DataFrame({"_partial_fname":[str(x).rsplit("_",4)[0] for x in fnames], "fname": fnames})
    df = df.merge(_tmp_merge_df, on="_partial_fname").drop(columns=["_partial_fname"])
    
    # Minor cleanup of our temporary workaround
    del _tmp_merge_df; gc.collect(); gc.collect()
    
    # 5. Get slice dimensions from filepath (int in pixels)
    df["slice_w"] = df["fname"].apply(lambda x: int(str(x)[:-4].rsplit("_",4)[1]))
    df["slice_h"] = df["fname"].apply(lambda x: int(str(x)[:-4].rsplit("_",4)[2]))

    # 6. Pixel spacing from filepath (float in mm)
    df["px_spacing_h"] = df["fname"].apply(lambda x: float(str(x)[:-4].rsplit("_",4)[3]))
    df["px_spacing_w"] = df["fname"].apply(lambda x: float(str(x)[:-4].rsplit("_",4)[4]))

    # 7. Merge 3 Rows Into A Single Row (As This/Segmentation-RLE Is The Only Unique Information Across Those Rows)
    l_bowel_train_df = df[df["class"]=="large_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"lb_seg_rle"})
    s_bowel_train_df = df[df["class"]=="small_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"sb_seg_rle"})
    stomach_train_df = df[df["class"]=="stomach"][["id", "segmentation"]].rename(columns={"segmentation":"st_seg_rle"})
    df = df.merge(l_bowel_train_df, on="id", how="left")
    df = df.merge(s_bowel_train_df, on="id", how="left")
    df = df.merge(stomach_train_df, on="id", how="left")
    df = df.drop_duplicates(subset=["id",]).reset_index(drop=True)
    df["lb_seg_flag"] = df["lb_seg_rle"].apply(lambda x: not pd.isna(x))
    df["sb_seg_flag"] = df["sb_seg_rle"].apply(lambda x: not pd.isna(x))
    df["st_seg_flag"] = df["st_seg_rle"].apply(lambda x: not pd.isna(x))
    df["n_segs"] = df["lb_seg_flag"].astype(int)+df["sb_seg_flag"].astype(int)+df["st_seg_flag"].astype(int)
    
    df = df.sort_values(by=['id']).copy()
    
    # Add 2.5D fnames
    for j, i in enumerate(range(-1*(channels-channels//2-1), channels//2+1)):
        method = 'ffill'
        if i <= 0: method = 'bfill'
        df[f'fname_{j:02}'] = df.groupby(['case_id', 'day_num'])['fname'].shift(stride*-i).fillna(method=method)
        # df['fname_01'] = df.groupby(['case_id', 'day_num'])['fname'].shift(1*stride).fillna(method='bfill')
        # df['fname_02'] = df.groupby(['case_id', 'day_num'])['fname'].shift(0*stride).fillna(method='bfill')
        # df['fname_03'] = df.groupby(['case_id', 'day_num'])['fname'].shift(-1*stride).fillna(method='ffill')
        # df['fname_04'] = df.groupby(['case_id', 'day_num'])['fname'].shift(-2*stride).fillna(method='ffill')
    
    # for i in range(channels):
    #     df[f'fname_{i:02}'] = df.groupby(['case_id','day_num'])['fname'].shift(-i*stride).fillna(method="ffill")
    df['fnames'] = df[[f'fname_{j:02d}' for j in range(channels)]].values.tolist()

    # 8. Reorder columns to the a new ordering (drops class and segmentation as no longer necessary)
    df = df[["id", "fname", "fnames", "n_segs",
             "lb_seg_rle", "lb_seg_flag",
             "sb_seg_rle", "sb_seg_flag", 
             "st_seg_rle", "st_seg_flag",
             "slice_h", "slice_w", "px_spacing_h", 
             "px_spacing_w", "case_id_str", "case_id", 
             "day_num_str", "day_num", "case_id_day_num_str", "slice_id",]].reset_index(drop=True)
    

    return df

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# modified from: https://www.kaggle.com/inversion/run-length-decoding-quick-start
def rle_decode(mask_rle, shape, color=1):
    """ TBD
    
    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return 
    
    Returns: 
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background

    """
    # Split the string by space, then convert it into a integer array
    s = np.array(mask_rle.split(), dtype=int)

    # Every even value is the start, every odd value is the "run" length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    # The image image is actually flattened since RLE is a 1D "run"
    if len(shape)==3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    # The color here is actually just any integer you want!
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
        
    # Don't forget to change the image back to the original shape
    return img.reshape(shape)

def pad_img(img, up_size=None):
    if up_size is None:
        return img
    shape0 = np.array(img.shape[:2])
    resize = np.array(up_size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        img = np.pad(img, [pady, padx])
        img = img.reshape((*resize))
    return img


def unpad_img(img, up_size, org_size):
    shape0 = np.array(org_size)
    resize = np.array(up_size)
    if np.any(shape0!=resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pady = [pad0//2, pad0//2 + pad0%2]
        padx = [pad1//2, pad1//2 + pad1%2]
        img = img[pady[0]:-pady[1], padx[0]:-padx[1], :]
        img = img.reshape((*shape0, 3))
    return img

def get_image(row, up_size=None):
    img = np.array(Image.open(row['fname']))
    img = np.interp(img, [np.min(img), np.max(img)], [0,255])
    return pad_img(img, up_size)
    # return row['fname']
    
def load_image(fname, up_size=None):
    img = np.array(Image.open(fname))
    img = np.interp(img, [np.min(img), np.max(img)], [0,255])
    return pad_img(img, up_size)

def get_25D_image(row, up_size=None):
    if up_size:
        imgs = np.zeros((*up_size, len(row['fnames'])))
    else:
        imgs = np.zeros((row['slice_h'], row['slice_w'], len(row['fnames'])))
        
    for i, fname in enumerate(row['fnames']):
        img = load_image(fname, up_size)
        imgs[..., i] += img
    return imgs.astype(np.uint8)
                   

def get_mask(row, up_size=None):
    if up_size:
        mask = np.zeros((*up_size, 3))
    else:
        mask = np.zeros((row['slice_h'], row['slice_w'], 3))
        
    if row['lb_seg_flag']:
        mask[..., 0] += pad_img(rle_decode(row['lb_seg_rle'], shape=(row['slice_h'], row['slice_w']), color=255), up_size)
    if row['sb_seg_flag']:
        mask[..., 1] += pad_img(rle_decode(row['sb_seg_rle'], shape=(row['slice_h'], row['slice_w']), color=255), up_size)
    if row['st_seg_flag']:
        mask[..., 2] += pad_img(rle_decode(row['st_seg_rle'], shape=(row['slice_h'], row['slice_w']), color=255), up_size)
        
    return mask.astype(np.uint8)

def get_targs(row): return row[['lb_seg_flag', 'sb_seg_flag', 'st_seg_flag']].values.astype(np.uint8)

def add_custom_valid(train, val_pct, seed):
    np.random.seed(seed)

    cases = train.case_id.unique()
    n_cases = len(cases)
    random_cases = np.random.choice(cases, int(n_cases*val_pct/1.85), replace=False)

    train['is_valid'] = False
    train.loc[train.case_id.isin(random_cases), 'is_valid'] = True
    
    days = train.loc[~train['is_valid'], 'case_id_day_num_str'].unique()
    n_days = len(days)
    random_days = np.random.choice(days, int(n_days*val_pct/1.85), replace=False)

    train.loc[train.case_id_day_num_str.isin(random_days), 'is_valid'] = True
    
    return train

def add_group_valid(train, val_pct, seed, fold=0):
    gss = GroupShuffleSplit(n_splits=5, test_size=val_pct, random_state=seed)
    train_idx, val_idx = [(train_idx, val_idx) for (train_idx, val_idx) in gss.split(train, train, train['case_id'])][fold]
 
    train['is_valid'] = False
    train.loc[val_idx, 'is_valid'] = True
    
    return train



## Customize Datablock API

@ToTensor
def encodes(self, o:PILMask): return o._tensor_cls(image2tensor(o))

@Normalize
def encodes(self, o:TensorMask): return o / 255

@Normalize
def decodes(self, o:TensorMask): 
    f = to_cpu if o.device.type=='cpu' else noop
    return f((o * 255).long())

        
        
## DLS functions
@ToTensor
def encodes(self, o:np.ndarray): return TensorImage(image2tensor(o))


class SegmentationAlbumentationsTransform5C(ItemTransform, RandTransform):
    split_idx, order = None, 2
    def __init__(self, train_aug, valid_aug): store_attr()
    
    def before_call(self, b, split_idx): self.idx = split_idx
    
    def encodes(self, x):
        if len(x) > 1:
            img, mask = x
            if self.idx == 0:
                aug = self.train_aug(image=np.array(img), mask=np.array(mask))    
            else:
                aug = self.valid_aug(image=np.array(img), mask=np.array(mask))
            return aug["image"], PILMask.create(aug["mask"])
        else:
            img = x[0]
            aug = self.valid_aug(image=np.array(img))
            return aug["image"]


def get_5C_25D_dls(df, up_size=(320, 384), resize=0.5, channels=5, crop=0.9, val_crop=0.9, bs=16, sample=False, 
                   sample_empty=False, frac=0.2, empty_frac=0.1, aug_p=0.5, val_pct=0.2, show=True, aug='fastai',
                   tfms_kwargs=dict(), val='group', fold=0, seed=42):
    
    df = df.copy()
    
    np.random.seed(seed)
    set_seed(seed, True)
    
    if val == 'group':
        df = add_group_valid(df, val_pct, seed, fold)
    elif val == 'custom':
        df = add_custom_valid(df, val_pct, seed)
    
        
    if sample_empty:
        df = pd.concat([
            df.query('n_segs == 0 & is_valid == False').sample(frac=empty_frac, replace=False, random_state=seed),
            df.query('n_segs != 0 | is_valid == True')
        ])
    
    if sample:
        dev = df.sample(frac=frac, random_state=seed)
    else:
        dev = df.sample(frac=1.0, random_state=seed)
        
                
    tfms = [[partial(get_25D_image, up_size=up_size)],
            [partial(get_mask, up_size=up_size), PILMask.create]]
    
    # after_item = [
                  # Resize(size=up_size, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
                  # Resize(size=[round(crop*size) for size in up_size]),
                  # SegmentationAlbumentationsTransform5C(get_train_aug(up_size, crop=crop, p=aug_p), 
                  #                                       get_test_aug(up_size, crop=val_crop)),
                  # ToTensor()]
    
    after_item = [ToTensor()]
    
    if aug == 'fastai':
        
        after_item = [SegmentationAlbumentationsTransform5C(get_train_crop(up_size, crop=crop, p=aug_p), 
                                                            get_test_crop(up_size, crop=val_crop))] + after_item
        
    else:
        
        after_item = [SegmentationAlbumentationsTransform5C(get_train_aug(up_size, resize=resize, crop=crop, p=aug_p), 
                                                            get_test_aug(up_size, resize=resize, crop=val_crop))] + after_item
    
    
    # after_item = [ToTensor()]
    
    # tfms = [[get_25D_image, partial(cv2.resize, dsize=img_size)], [get_mask, partial(cv2.resize, dsize=img_size), PILMask.create]]
    # after_item = [SegmentationAlbumentationsTransform5C(get_train_aug(img_size), get_test_aug(img_size)), ToTensor()]

    splits = ColSplitter()(dev)
    dsets = Datasets(dev, tfms, splits=splits)
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    if channels == 5:
        means = means + means[:channels-3]
        stds = stds + stds[:channels-3]
        
    if channels == 7:
        means = means + means + means[:1]
        stds = stds + stds + stds[:1]
    
    if aug == 'fastai':
        
        if resize < 1.0:
            size = round(resize*crop*up_size[0])
        else:
            size = None
    
        batch_tfms = [IntToFloatTensor, 
                      *aug_transforms(size=None, **tfms_kwargs), 
                      Normalize.from_stats(means, stds)]
    else:
        batch_tfms = [IntToFloatTensor, 
                      # *aug_transforms(size=round(resize*crop*up_size[0])), 
                      Normalize.from_stats(means, stds)]

    # if resize < 1:
    #     img_size = [round(crop*size*resize) for size in up_size]
    #     batch_tfms = [ResizeBatch(img_size)] + batch_tfms
    
    dls = dsets.dataloaders(bs=bs, after_item=after_item,
                           after_batch=batch_tfms)
    dls.rng.seed(seed)
    
    if show:
        nrows = bs//4
        ncols = 4
        dls.show_batch(nrows=nrows, ncols=ncols, max_n=bs, figsize=(ncols*3, nrows*3))
        
    return dls, dev


@typedispatch
def show_batch(x:TensorImage, y:TensorMask, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*3, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(max_n, nrows=nrows, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs): 
        x_i = x[i] / x[i].max()
        md_ch = x_i.shape[0] // 2
        show_image(x_i[[md_ch-1, md_ch, md_ch+1]], ctx=ctx, cmap='gray', **kwargs)
        show_image(y[i], ctx=ctx, cmap='Spectral_r', alpha=0.35, **kwargs)
        red_patch = mpatches.Patch(color='red', label='lb')
        green_patch = mpatches.Patch(color='green', label='sb')
        blue_patch = mpatches.Patch(color='blue', label='st')
        ctx.legend(handles=[red_patch, green_patch, blue_patch], fontsize=figsize[0]/2)

class SegmentationAlbumentationsTransform(ItemTransform):
    def __init__(self, aug, split_idx): store_attr()
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])
    
class AlbumentationsTransform(DisplayedTransform):
    order = 2
    def __init__(self, aug, split_idx): store_attr()
    def encodes(self, x: PILImage):
        aug = self.aug(image=np.array(x))
        return PILImage.create(aug["image"])
    

def get_train_aug(img_size):
    if isinstance(img_size, int): img_size = (img_size, img_size)
    return albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            albumentations.OneOf([
            albumentations.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            albumentations.CoarseDropout(max_holes=8, max_height=img_size[0]//20, max_width=img_size[1]//20,
                             min_holes=5, fill_value=0, mask_fill_value=0, p=0.5)
    ])


def get_train_crop(img_size, crop=0.9, p=0.4):
    crop_size = round(crop*img_size[0])
    return albumentations.Compose([
            albumentations.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
            ])


def get_test_crop(img_size, crop=0.9):
    crop_size = round(crop*img_size[0])
    return  albumentations.Compose([
        albumentations.CenterCrop(height=crop_size, width=crop_size),
    ])

def get_dls(df, img_size=224, method='squish', bs=16, sample=False, sample_empty=False, frac=0.2, val_pct=0.2, show=True, val='group', seed=42):
    
    df = df.copy()
    
    np.random.seed(seed)
    set_seed(seed, True)
    
    if val == 'group':
        df = add_group_valid(df, val_pct)
    elif val == 'custom':
        df = add_custom_valid(df, val_pct)
    
        
    if sample_empty:
        df = pd.concat([
            df.query('n_segs == 0 & is_valid == False').sample(frac=0.1, replace=False, random_state=seed),
            df.query('n_segs != 0 | is_valid == True')
        ])
    
    if sample:
        dev = df.sample(frac=frac, random_state=seed)
    else:
        dev = df.sample(frac=1.0, random_state=seed)
    
    tfms = [[get_image, PILImageBW.create], [get_mask, PILMask.create]]
    splits = ColSplitter()(dev)
    dsets = Datasets(dev, tfms, splits=splits)
    dls = dsets.dataloaders(bs=bs, after_item=[Resize(img_size, method=method),
                                               SegmentationAlbumentationsTransform(get_train_aug(img_size), 0),
                                               ToTensor()],
                           after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
    dls.rng.seed(seed)
    
    if show:
        dls.show_batch(nrows=bs//4, ncols=4, max_n=bs, figsize=(12, 12))
        
    return dls, dev

def get_aug_dls(train, aug=[], img_size=224, method='squish', bs=16, sample=False, show=True, val='group', seed=42):
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]
    if aug: batch_tfms = [*aug] + batch_tfms
    
    db = DataBlock((ImageBlock(cls=PILImageBW), MaskBlock),
                   get_x=get_image,
                   get_y=get_mask,
                   splitter = ColSplitter(),
                   item_tfms=[Resize(img_size, method=method)],
                   batch_tfms=batch_tfms)
    
    if val == 'group':
        train = add_group_valid(train, seed)
    elif val == 'custom':
        train = add_custom_valid(train, seed)
    
    if sample:
        dev = train.sample(frac=0.2, random_state=seed)
    else:
        dev = train
        
    dls = db.dataloaders(dev, bs=bs, shuffle=True)
    dls.rng.seed(seed)
    
    if show:
        dls.show_batch(nrows=bs//4, ncols=4, max_n=bs, figsize=(12, 12))
        
    return dls, dev

def get_25D_dls(df, img_size=224, method='squish', bs=16, sample=False, sample_empty=False, frac=0.2, val_pct=0.2, show=True, val='group', seed=42):
    
    df = df.copy()
    
    np.random.seed(seed)
    set_seed(seed, True)
    
    if val == 'group':
        df = add_group_valid(df, val_pct, seed)
    elif val == 'custom':
        df = add_custom_valid(df, val_pct, seed)
    
        
    if sample_empty:
        df = pd.concat([
            df.query('n_segs == 0 & is_valid == False').sample(frac=0.1, replace=False, random_state=seed),
            df.query('n_segs != 0 | is_valid == True')
        ])
    
    if sample:
        dev = df.sample(frac=frac, random_state=seed)
    else:
        dev = df.sample(frac=1.0, random_state=seed)
        
    if isinstance(img_size, int): img_size = (img_size, img_size)
        
    if img_size[0] > 310 or img_size[1] > 360:
        tfms = [[partial(get_25D_image, up_size=img_size), PILImageBW.create], [partial(get_mask, up_size=img_size), PILMask.create]]
        after_item = [SegmentationAlbumentationsTransform(get_train_aug(img_size), 0), ToTensor()]
    else:
        tfms = [[get_25D_image, PILImage.create], [get_mask, PILMask.create]]
        after_item = [Resize(img_size, method=method), SegmentationAlbumentationsTransform(get_train_aug(img_size), 0), ToTensor()]
    
    splits = ColSplitter()(dev)
    dsets = Datasets(dev, tfms, splits=splits)
    dls = dsets.dataloaders(bs=bs, after_item=after_item,
                           after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
    dls.rng.seed(seed)
    
    if show:
        nrows = bs//4
        ncols = 4
        dls.show_batch(nrows=nrows, ncols=ncols, max_n=bs, figsize=(ncols*3, nrows*3))
        
    return dls, dev


## Metrics
from scipy.spatial.distance import directed_hausdorff

def mod_acc(inp, targ):
    targ = targ.squeeze(1)
    mask = targ != 0
    if mask.sum() == 0:
        mask = targ == 0
    return (torch.where(sigmoid(inp) > 0.5, 1, 0)[mask]==targ[mask]).float().mean().item()

def dice_coeff_old(inp, targ):
    inp = np.where(inp.cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    eps = 1e-5
    I = (targ * inp).sum((2, 3))
    U =  targ.sum((2,3)) + inp.sum((2, 3))
    return ((2.*I+eps)/(U+eps)).mean((1, 0))

# def dice_coeff(inp, targ):
#     if torch.is_tensor(inp):
#         inp = torch.where(sigmoid(inp) > 0.5, 1, 0).cpu().detach().numpy().astype(np.uint8)
#     if torch.is_tensor(targ):
#         targ = targ.cpu().detach().numpy().astype(np.uint8)
#     # mask = targ == 1
#     # I = (inp[mask] == targ[mask]).sum((2, 3))
#     eps = 1e-5
#     I = (targ & inp).sum((2, 3))
#     # U = inp.sum((2, 3)) + targ.sum((2, 3))
#     U = (targ | inp).sum((2, 3))
#     return ((2*I)/(U+I+1) + (U==0)).mean((1, 0))

# def dice_coeff2(inp, targ, thr=0.5, dim=(2,3), epsilon=0.001):
#     targ = targ.to(torch.float32)
#     inp = (inp>thr).to(torch.float32)
#     inter = (targ*inp).sum(dim=dim)
#     den = targ.sum(dim=dim) + inp.sum(dim=dim)
#     dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
#     return dice

def dice_coeff(inp, targ):
    inp = np.where(inp.cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    eps = 1e-5
    dice_scores = []
    for i in range(targ.shape[0]):
        dice_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            I = (targ[i, j] * inp[i, j]).sum()
            U =  targ[i, j].sum() + inp[i, j].sum()
            dice_i.append((2.*I)/(U+eps))
        if dice_i:
            dice_scores.append(np.mean(dice_i))
    
    if dice_scores:
        return np.mean(dice_scores)
    else:
        return 0

def hd_dist_per_slice(inp, targ, seed=42):    
    inp = np.argwhere(inp) / np.array(inp.shape)
    targ = np.argwhere(targ) / np.array(targ.shape)
    # if len(targ) == 0:
    #     inp = 1 - inp
    #     targ = 1 - targ
    haussdorf_dist = 1 - directed_hausdorff(inp, targ, seed)[0]
    return haussdorf_dist if haussdorf_dist > 0 else 0

def hd_dist_old(inp, targ, seed):
    inp = np.where(inp.cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    
    return np.mean([np.mean([hd_dist_per_slice(inp[i, j], targ[i, j], seed) for j in range(3)]) for i in range(len(inp))])

def hd_dist(inp, targ, seed):
    inp = np.where(inp.cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    
    hd_scores = []
    for i in range(targ.shape[0]):
        hd_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            hd_i.append(hd_dist_per_slice(inp[i, j], targ[i, j], seed))
        if hd_i:
            hd_scores.append(np.mean(hd_i))
    if hd_scores:
        return np.mean(hd_scores)
    else:
        return 0

def custom_metric(inp, targ, seed=42):
    hd_score_per_batch = hd_dist(inp, targ, seed)
    dice_score_per_batch = dice_coeff(inp, targ)
    
    return 0.4*dice_score_per_batch + 0.6*hd_score_per_batch

def custom_metric_old(inp, targ, seed=42):
    hd_score_per_batch = hd_dist_old(inp, targ, seed)
    dice_score_per_batch = dice_coeff_old(inp, targ)
    
    return 0.4*dice_score_per_batch + 0.6*hd_score_per_batch

def post_process(inp):
    if inp.shape[-1] != 3: 
        sum_dims = (2, 3)
    else:
        sum_dims = (1, 2)
    inp[np.where(inp.sum(sum_dims) < [11, 29, 12])] = 0 
    return inp

def dice_coeff_adj(inp, targ):
    inp = np.where(sigmoid(inp).cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    inp = post_process(inp)
    eps = 1e-5
    dice_scores = []
    for i in range(targ.shape[0]):
        dice_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            I = (targ[i, j] * inp[i, j]).sum()
            U =  targ[i, j].sum() + inp[i, j].sum()
            dice_i.append((2.*I)/(U+eps))
        if dice_i:
            dice_scores.append(np.mean(dice_i))
    
    if dice_scores:
        return np.mean(dice_scores)
    else:
        return 0
    
    
def hd_dist_per_slice(inp, targ, seed):    
    inp = np.argwhere(inp) / np.array(inp.shape)
    targ = np.argwhere(targ) / np.array(targ.shape)
    # if len(targ) == 0:
    #     inp = 1 - inp
    #     targ = 1 - targ
    haussdorf_dist = 1 - directed_hausdorff(inp, targ, seed)[0]
    return haussdorf_dist if haussdorf_dist > 0 else 0

def hd_dist_adj(inp, targ, seed=42):
    inp = np.where(sigmoid(inp).cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    inp = post_process(inp)
    hd_scores = []
    for i in range(targ.shape[0]):
        hd_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            hd_i.append(hd_dist_per_slice(inp[i, j], targ[i, j], seed))
        if hd_i:
            hd_scores.append(np.mean(hd_i))
    if hd_scores:
        return np.mean(hd_scores)
    else:
        return 0

def custom_metric_adj(inp, targ, seed=42):
    hd_score_per_batch = hd_dist_adj(inp, targ, seed)
    dice_score_per_batch = dice_coeff_adj(inp, targ)
    
    return 0.4*dice_score_per_batch + 0.6*hd_score_per_batch

class SkipEmptyMetric(AvgMetric):        
    def accumulate(self, learn):
        score, bs = self.func(learn.pred, *learn.yb)
        if score is not None:
            self.total += learn.to_detach(score)*bs
            self.count += bs
        else:
            print('score is none')
            
def dice_coeff_adj_skip(inp, targ):
    inp = np.where(inp.cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    inp = post_process(inp)
    eps = 1e-5
    dice_scores = []
    for i in range(targ.shape[0]):
        dice_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            I = (targ[i, j] * inp[i, j]).sum()
            U =  targ[i, j].sum() + inp[i, j].sum()
            dice_i.append((2.*I)/(U+eps))
        if dice_i:
            dice_scores.append(np.mean(dice_i))
    
    if dice_scores:
        return np.mean(dice_scores), len(dice_scores)
    else:
        return None
    
    
def hd_dist_adj_skip(inp, targ, seed=42):
    inp = np.where(inp.cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    inp = post_process(inp)
    hd_scores = []
    for i in range(targ.shape[0]):
        hd_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            hd_i.append(hd_dist_per_slice(inp[i, j], targ[i, j], seed))
        if hd_i:
            hd_scores.append(np.mean(hd_i))
    if hd_scores:
        return np.mean(hd_scores), len(hd_scores)
    else:
        return None


# Loss functions
class DiceBCEModule(Module):
    def __init__(self, eps:float=1e-5, from_logits=True):
        store_attr()
        
    def forward(self, inp:Tensor, targ:Tensor) -> Tensor:
        inp = inp.view(-1)
        targ = targ.view(-1)
        
        if self.from_logits: 
            bce_loss = nn.BCEWithLogitsLoss()(inp, targ)
            inp = torch.sigmoid(inp)
            
            
        intersection = (inp * targ).sum()                            
        dice = (2.*intersection + self.eps)/(inp.sum() + targ.sum() + self.eps)  
        
        return 0.5*(1 - dice) + 0.5*bce_loss


class DiceBCELoss(BaseLoss):
    def __init__(self, *args, eps:float=1e-5, from_logits=True, thresh=0.5, **kwargs):
        super().__init__(DiceBCEModule, *args, eps=eps, from_logits=from_logits, flatten=False, is_2d=True, floatify=True, **kwargs)
        self.thresh = thresh
    
    def decodes(self, x:Tensor) -> Tensor:
        "Converts model output to target format"
        return (x>self.thresh).long()

    def activation(self, x:Tensor) -> Tensor:
        "`nn.BCEWithLogitsLoss`'s fused activation function applied to model output"
        return torch.sigmoid(x)

# Source: https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained/notebook
def focal_binary_cross_entropy(logits, targets, gamma=2, n=3):
    p = torch.sigmoid(logits)
    p = torch.where(targets >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = n*loss.mean()
    return loss

class DiceFocalModule(Module):
    def __init__(self, eps:float=1e-5, from_logits=True, ws=[0.5, 0.5], gamma=2, n=3):
        store_attr()
        
    def forward(self, inp:Tensor, targ:Tensor) -> Tensor:
        inp = inp.view(-1)
        targ = targ.view(-1)
        
        if self.from_logits: 
            focal_loss = focal_binary_cross_entropy(inp, targ, self.gamma, self.n)
            inp = torch.sigmoid(inp)
            
            
        intersection = (inp * targ).sum()                            
        dice = (2.*intersection + self.eps)/(inp.sum() + targ.sum() + self.eps)  
        
        return self.ws[0]*(1 - dice) + self.ws[1]*focal_loss
    
class DiceFocalLoss(BaseLoss):
    def __init__(self, *args, eps:float=1e-5, from_logits=True, ws=[0.5, 0.5], gamma=2, n=3, thresh=0.5, **kwargs):
        super().__init__(DiceFocalModule, *args, eps=eps, from_logits=from_logits, ws=ws, gamma=gamma, n=n, flatten=False, is_2d=True, floatify=True, **kwargs)
        self.thresh = thresh
    
    def decodes(self, x:Tensor) -> Tensor:
        "Converts model output to target format"
        return (x>self.thresh).long()

    def activation(self, x:Tensor) -> Tensor:
        "`nn.BCEWithLogitsLoss`'s fused activation function applied to model output"
        return torch.sigmoid(x)

class FocalTverskyLossModule(Module):
    def __init__(self, eps:float=1e-5, from_logits=True, alpha=0.3, beta=0.7, gamma=3/4):
        store_attr()
        
    def forward(self, inp:Tensor, targ:Tensor) -> Tensor:
        inp = inp.view(-1)
        targ = targ.view(-1)
        
        if self.from_logits: 
            inp = torch.sigmoid(inp)
            
        inp_0, inp_1 = inp, 1 - inp
        targ_0, targ_1 = targ, 1 - targ
            
        num = (inp_0 * targ_0).sum() 
        denom = num + (self.alpha * (inp_0 * targ_1).sum()) + (self.beta * (inp_1 * targ_0).sum()) + self.eps
        loss = 1 - (num / denom)
        return loss**self.gamma 
    
class FocalTverskyLoss(BaseLoss):
    def __init__(self, *args, eps:float=1e-5, from_logits=True, alpha=0.3, beta=0.7, gamma=3/4, thresh=0.5, **kwargs):
        super().__init__(FocalTverskyLossModule, *args, eps=eps, from_logits=from_logits, alpha=alpha, beta=beta, gamma=gamma, flatten=False, is_2d=True, floatify=True, **kwargs)
        self.thresh = thresh
    
    def decodes(self, x:Tensor) -> Tensor:
        "Converts model output to target format"
        return (x>self.thresh).long()

    def activation(self, x:Tensor) -> Tensor:
        "`nn.BCEWithLogitsLoss`'s fused activation function applied to model output"
        return torch.sigmoid(x)

def focal_binary_cross_entropy(logits, targets, gamma=2, n=3):
    p = torch.sigmoid(logits)
    p = torch.where(targets >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = n*loss.mean()
    return loss

class ComboModule(Module):
    def __init__(self, eps:float=1e-5, from_logits=True, ws=[2, 3, 1], gamma=2, n=3):
        store_attr()
        
    def forward(self, inp:Tensor, targ:Tensor) -> Tensor:
        inp = inp.view(-1)
        targ = targ.view(-1)
        
        if self.from_logits: 
            focal_loss = focal_binary_cross_entropy(inp, targ, self.gamma, self.n)
            bce_loss = nn.BCEWithLogitsLoss()(inp, targ)
            inp = torch.sigmoid(inp)
                
        intersection = (inp * targ).sum()                            
        dice = (2.*intersection + self.eps)/(inp.sum() + targ.sum() + self.eps)  
        
        return self.ws[0]*(1 - dice) + self.ws[1]*focal_loss + self.ws[2]*bce_loss
    
class ComboLoss(BaseLoss):
    def __init__(self, *args, eps:float=1e-5, from_logits=True, ws=[2, 3, 1], gamma=2, n=3, thresh=0.5, **kwargs):
        super().__init__(ComboModule, *args, eps=eps, from_logits=from_logits, ws=ws, gamma=gamma, n=n, flatten=False, is_2d=True, floatify=True, **kwargs)
        self.thresh = thresh
    
    def decodes(self, x:Tensor) -> Tensor:
        "Converts model output to target format"
        return (x>self.thresh).long()

    def activation(self, x:Tensor) -> Tensor:
        "`nn.BCEWithLogitsLoss`'s fused activation function applied to model output"
        return torch.sigmoid(x)
    
class DiceLossSMP(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(smp.losses.DiceLoss, *args, mode='multilabel', flatten=False, is_2d=True, floatify=True, **kwargs)
        
        
class TverskyLoss(BaseLoss):
    def __init__(self, *args, mode='multilabel', thresh=0.5, **kwargs):
        super().__init__(smp.losses.TverskyLoss, *args, mode=mode, flatten=False, is_2d=True, floatify=True, **kwargs)
        self.thresh = thresh
    
    def decodes(self, x:Tensor) -> Tensor:
        "Converts model output to target format"
        return (x>self.thresh).long()

    def activation(self, x:Tensor) -> Tensor:
        "`nn.BCEWithLogitsLoss`'s fused activation function applied to model output"
        return torch.sigmoid(x)

    
def deep_supervision_loss(inp, targ, loss_func=ComboLoss()):
    loss = 0
    for i in range(len(inp)):
        loss += loss_func(inp[i], targ)
    return loss/len(inp)    

# Model and learner
def build_model(encoder_name, in_c=3, classes=3, weights="imagenet"):
    model = smp.Unet(
        encoder_name=encoder_name,      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=weights,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=in_c,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=classes,        # model output channels (number of classes in your dataset)
        activation=None,
    )
    model.to('cuda')
    return model


def unet_splitter(model):
    model_layers = list(model.children())
    encoder_params = params(model_layers[0])
    decoder_params = params(model_layers[1]) + params(model_layers[2])
    return L(encoder_params, decoder_params)

def dynamic_unet_splitter(model):
    return L(model[0], model[1:]).map(params)

def cat_splitter(model): return L(model.body, model.head).map(params)

def get_learner(dls, arch, loss_func, seed=42, **kwargs):
    set_seed(seed, True)
    learn = unet_learner(dls, arch, metrics=[mod_acc, dice_coeff, hd_dist, custom_metric], n_out=3, loss_func=loss_func, **kwargs).to_fp16()
    return learn

def get_custom_learner(dls, model, loss_func, splitter, metrics=[dice_coeff_adj, hd_dist_adj, custom_metric_adj], seed=42, **kwargs):
    set_seed(seed, True)
    learn = Learner(dls, model, metrics=metrics, loss_func=loss_func, splitter=splitter, **kwargs).to_fp16()
    learn.freeze()
    return learn


## Nested Unets and functions
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final4(x0_4)
            return output

def deep_supervision_loss(inp, targ):
    loss = 0
    loss_func = ComboLoss()
    for i in range(len(inp)):
        loss += loss_func(inp[i], targ)
    return loss/len(inp)

def dice_coeff_deep(inp, targ):
    inp = np.where(sigmoid(inp[-1]).cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    inp = post_process(inp)
    eps = 1e-5
    dice_scores = []
    for i in range(targ.shape[0]):
        dice_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            I = (targ[i, j] * inp[i, j]).sum()
            U =  targ[i, j].sum() + inp[i, j].sum()
            dice_i.append((2.*I)/(U+eps))
        if dice_i:
            dice_scores.append(np.mean(dice_i))
    
    if dice_scores:
        return np.mean(dice_scores)
    else:
        return 0
    
    
def hd_dist_per_slice(inp, targ, seed=42):    
    inp = np.argwhere(inp) / np.array(inp.shape)
    targ = np.argwhere(targ) / np.array(targ.shape)
    # if len(targ) == 0:
    #     inp = 1 - inp
    #     targ = 1 - targ
    haussdorf_dist = 1 - directed_hausdorff(inp, targ, seed)[0]
    return haussdorf_dist if haussdorf_dist > 0 else 0

def hd_dist_deep(inp, targ):
    inp = np.where(sigmoid(inp[-1].cpu()).detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    inp = post_process(inp)
    hd_scores = []
    for i in range(targ.shape[0]):
        hd_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            hd_i.append(hd_dist_per_slice(inp[i, j], targ[i, j]))
        if hd_i:
            hd_scores.append(np.mean(hd_i))
    if hd_scores:
        return np.mean(hd_scores)
    else:
        return 0

def custom_metric_deep(inp, targ):
    hd_score_per_batch = hd_dist_deep(inp, targ)
    dice_score_per_batch = dice_coeff_deep(inp, targ)
    
    return 0.4*dice_score_per_batch + 0.6*hd_score_per_batch

def nested_splitter(model): return L(model).map(params)


# Debugging
def plt_before_after(idx):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    print(before_dices[idx], after_dices[idx], val.iloc[idx]['slice_w'], val.iloc[idx]['slice_h'])
    
    # plot image and target before upsize
    decoded_small_img, decoded_small_mask = dls.decode((imgs[idx], targs[idx]))
    show_image(decoded_small_img[0], cmap='gray', ctx=axes[0])
    show_image(decoded_small_mask, cmap='Spectral_r', alpha=0.35, ctx=axes[0], title="Target (resized)")
    
    # plot image and pred before upsize
    show_image(decoded_small_img[0], cmap='gray', ctx=axes[1])
    show_image(preds_masks[idx]*255, cmap='Spectral_r', alpha=0.35, ctx=axes[1], title="Prediction (resized)")
    
    # plot image and target after upsize
    show_image(org_imgs[idx], ctx=axes[2])
    show_image(targ_masks[idx], cmap='Spectral_r', alpha=0.35, ctx=axes[2], title="Target (original)")
    
    # plot image and pred after upsize
    show_image(org_imgs[idx], ctx=axes[3])
    show_image(np.moveaxis(inp_masks[idx], 0, -1)*255, cmap='Spectral_r', alpha=0.35, ctx=axes[3], title="Prediction (upsized)")


# Dyanmic unet for timm
import timm

def timm_model_sizes(encoder, img_size):
    sizes = []
    for layer in encoder.feature_info:
        sizes.append(torch.Size([1, layer['num_chs'], img_size[0]//layer['reduction'], img_size[1]//layer['reduction']]))
    return sizes


def get_timm_output_layers(encoder):
    outputs = []
    for layer in encoder.feature_info:
        # Converts 'blocks.0.0' to ['blocks', '0', '0']
        attrs = layer['module'].split('.')
        output_layer = getattr(encoder, attrs[0])[int(attrs[1])][int(attrs[2])]
        outputs.append(output_layer)
    return outputs


class DynamicTimmUnet(SequentialEx):
    "Create a U-Net from a given architecture in timm."
    def __init__(self, encoder, n_out, img_size, blur=False, blur_final=True, self_attention=False,
                 y_range=None, last_cross=True, bottle=False, act_cls=defaults.activation,
                 init=nn.init.kaiming_normal_, norm_type=None, **kwargs):
        imsize = img_size
        sizes = timm_model_sizes(encoder, img_size)
        sz_chg_idxs = list(reversed(range(len(sizes))))
        outputs = list(reversed(get_timm_output_layers(encoder)))
        self.sfs = hook_outputs(outputs, detach=False)
        
        # cut encoder
        encoder = nn.Sequential(*list(encoder.children()))[:-5]
        
        x = dummy_eval(encoder, imsize).detach()

        ni = sizes[-1][1]
        middle_conv = nn.Sequential(ConvLayer(ni, ni*2, act_cls=act_cls, norm_type=norm_type, **kwargs),
                                    ConvLayer(ni*2, ni, act_cls=act_cls, norm_type=norm_type, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, BatchNorm(ni), nn.ReLU(), middle_conv]

        for i,idx in enumerate(sz_chg_idxs):
            not_final = i!=len(sz_chg_idxs)-1
            up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i==len(sz_chg_idxs)-3)
            unet_block = UnetBlock(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=do_blur, self_attention=sa,
                                   act_cls=act_cls, init=init, norm_type=norm_type, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sizes[0][-2:]: layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))
        layers.append(ResizeToOrig())
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(ResBlock(1, ni, ni//2 if bottle else ni, act_cls=act_cls, norm_type=norm_type, **kwargs))
        layers += [ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=norm_type, **kwargs)]
        apply_init(nn.Sequential(layers[3], layers[-2]), init)
        #apply_init(nn.Sequential(layers[2]), init)
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        layers.append(ToTensorBase())
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()
        
        
class CatModel(nn.Module):
    def __init__(self, body, head):
        super(CatModel, self).__init__()
        self.body = body
        self.head = head
    def forward(self, x):
        return self.head(self.body(x))
    
## Unet#
class UNetBlock(nn.Module):
    def __init__(self, ni, nf, final_div=True, blur=False, act_cls=defaults.activation,
                 self_attention=False, init=nn.init.kaiming_normal_, norm_type=None, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.conv2 = ConvLayer(nf, nf, act_cls=act_cls, norm_type=norm_type,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = act_cls()
        apply_init(nn.Sequential(self.conv1, self.conv2), init)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))

class UNetDashDecoder(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        
        self.unet0_1 = UNetBlock(self.sizes[0]+self.sizes[1], self.sizes[0])
        self.unet1_1 = UNetBlock(self.sizes[1]+self.sizes[2], self.sizes[1])
        self.unet2_1 = UNetBlock(self.sizes[2]+self.sizes[3], self.sizes[2])
        self.unet3_1 = UNetBlock(self.sizes[3]+self.sizes[4], self.sizes[3])
        
        self.unet0_2 = UNetBlock(2*self.sizes[0]+self.sizes[1]+self.sizes[2], self.sizes[0])
        self.unet1_2 = UNetBlock(2*self.sizes[1]+self.sizes[2]+self.sizes[3], self.sizes[1])
        self.unet2_2 = UNetBlock(2*self.sizes[2]+self.sizes[3]+self.sizes[4], self.sizes[2])
        
        self.unet0_3 = UNetBlock(3*self.sizes[0]+self.sizes[1]+self.sizes[2]+self.sizes[3], self.sizes[0])
        self.unet1_3 = UNetBlock(3*self.sizes[1]+self.sizes[2]+self.sizes[3]+self.sizes[4], self.sizes[1])
        
        self.unet0_4 = UNetBlock(4*self.sizes[0]+self.sizes[1]+self.sizes[2]+self.sizes[3]+self.sizes[4], self.sizes[0])

        
        
    def forward(self, x0_0, x1_0, x2_0, x3_0, x4_0):
        x0_1 = self.unet0_1(torch.cat([x0_0, self.up2(x1_0)], 1))
        x1_1 = self.unet1_1(torch.cat([x1_0, self.up2(x2_0)], 1))
        x2_1 = self.unet2_1(torch.cat([x2_0, self.up2(x3_0)], 1))
        x3_1 = self.unet3_1(torch.cat([x3_0, self.up2(x4_0)], 1))
        
        x0_2 = self.unet0_2(torch.cat([x0_0, x0_1, self.up2(x1_1), self.up4(x2_0)], 1))
        x1_2 = self.unet1_2(torch.cat([x1_0, x1_1, self.up2(x2_1), self.up4(x3_0)], 1))
        x2_2 = self.unet2_2(torch.cat([x2_0, x2_1, self.up2(x3_1), self.up4(x4_0)], 1))
        
        x0_3 = self.unet0_3(torch.cat([x0_0, x0_1, x0_2, self.up2(x1_2), self.up4(x2_1), self.up8(x3_0)], 1))
        x1_3 = self.unet1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2), self.up4(x3_1), self.up8(x4_0)], 1))
                                            
        x0_4 = self.unet0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up2(x1_3), self.up4(x2_2), self.up8(x3_1), self.up16(x4_0)], 1))
        
        return x0_1, x0_2, x0_3, x3_1, x2_2, x1_3, x0_4
        
        

class UNetDash(nn.Module):
    def __init__(self, encoder, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.sizes = [size[1] for size in timm_model_sizes(encoder, (224, 224))]
        self.decoder = UNetDashDecoder(self.sizes)
        
        if self.deep_supervision:
            final_in = 4*self.sizes[0]+self.sizes[1]+self.sizes[2]+self.sizes[3]
        else:
            final_in = self.sizes[0]
        self.final = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), 
                                   ResBlock(1, final_in, final_in//2, ks=1),
                                   ConvLayer(final_in//2, final_in, ks=1, act_cls=None, norm_type=None),
                                   ConvLayer(final_in, 3, ks=1, act_cls=None, norm_type=None))     
        
    def forward(self, x):
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)
        x0_1, x0_2, x0_3, x3_1, x2_2, x1_3, x0_4 = self.decoder(x0_0, x1_0, x2_0, x3_0, x4_0)
        
        if self.deep_supervision:
            return x0_1, x0_2, x0_3, x3_1, x2_2, x1_3, x0_4
        else:
            return self.final(x0_4)
    
# Inference
import cv2

def dice_coeff_adj_pp(inp, targ):
    if torch.is_tensor(inp): inp = np.where(inp.cpu().detach().numpy() > 0.5, 1, 0)
    targ = targ.cpu().detach().numpy()
    inp = post_process(inp)
    eps = 1e-5
    dice_scores = []
    for i in range(targ.shape[0]):
        dice_i = []
        for j in range(targ.shape[1]):
            if inp[i, j].sum() == targ[i, j].sum() == 0:
                continue
            I = (targ[i, j] * inp[i, j]).sum()
            U =  targ[i, j].sum() + inp[i, j].sum()
            dice_i.append((2.*I)/(U+eps))
        if dice_i:
            dice_scores.append(np.mean(dice_i))
    
    if dice_scores:
        return np.mean(dice_scores)
    else:
        return 0

# Source: https://www.kaggle.com/code/clemchris/gi-seg-pytorch-train-infer

def mask2rle(mask):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    mask = np.array(mask)
    pixels = mask.flatten()
    pad = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def get_rle_masks(preds, df, up_size):
    rle_masks = []
    for pred, width, height in zip(preds, df['slice_w'], df['slice_h']):
        upsized_mask = unpad_img(pred, up_size, (width, height))
        for i in range(3):
            rle_mask = mask2rle(upsized_mask[:, :, i])
            rle_masks.append(rle_mask)
    return rle_masks


#### UnetPlusPlus
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.base import SegmentationHead

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    
class UNetPlusPlus(nn.Module):
    def __init__(self, model, in_channels, classes, decoder_channels=(256, 128, 64, 32, 16), pretrained=True, deep_supervision=True, **kwargs):
        super().__init__()
        
        self.encoder = timm.create_model(model, in_chans=in_channels, pretrained=pretrained, features_only=True)
        
        self.deep_supervision = deep_supervision
        
        encoder_channels = [size[1] for size in timm_model_sizes(self.encoder, (320, 384))]
        
        # computing blocks input and output channels
        head_channels = encoder_channels[-1]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        
        self.skip_channels = list(encoder_channels[::-1][1:]) + [0]
        self.out_channels = decoder_channels
        
        
        self.blocks = {}
        
        self.blocks['x_0_1'] = DecoderBlock(encoder_channels[1], encoder_channels[0], decoder_channels[-1])
        self.blocks['x_1_1'] = DecoderBlock(encoder_channels[2], encoder_channels[1], decoder_channels[-2])
        self.blocks['x_2_1'] = DecoderBlock(encoder_channels[3], encoder_channels[2], decoder_channels[-3])
        self.blocks['x_3_1'] = DecoderBlock(encoder_channels[4], encoder_channels[3], decoder_channels[-4])
        
        self.blocks['x_0_2'] = DecoderBlock(decoder_channels[-2], encoder_channels[0]+decoder_channels[-1], decoder_channels[-1])
        self.blocks['x_1_2'] = DecoderBlock(decoder_channels[-3], encoder_channels[1]+decoder_channels[-2], decoder_channels[-2])
        self.blocks['x_2_2'] = DecoderBlock(decoder_channels[-4], encoder_channels[2]+decoder_channels[-3], decoder_channels[-3])
        
        self.blocks['x_0_3'] = DecoderBlock(decoder_channels[-2], encoder_channels[0]+2*decoder_channels[-1], decoder_channels[-1])
        self.blocks['x_1_3'] = DecoderBlock(decoder_channels[-3], encoder_channels[1]+2*decoder_channels[-2], decoder_channels[-2])
        
        self.blocks['x_0_4'] = DecoderBlock(decoder_channels[-2], encoder_channels[0]+3*decoder_channels[-1], decoder_channels[-1])
        
        self.blocks['x_0_1_ex'] = DecoderBlock(decoder_channels[-1], 0, decoder_channels[-1])
        self.blocks['x_0_2_ex'] = DecoderBlock(decoder_channels[-1], decoder_channels[-1], decoder_channels[-1])
        self.blocks['x_0_3_ex'] = DecoderBlock(decoder_channels[-1], 2*decoder_channels[-1], decoder_channels[-1])
        self.blocks['x_0_4_ex'] = DecoderBlock(decoder_channels[-1], 3*decoder_channels[-1], decoder_channels[-1])
        
        for i in range(1, 5):
            self.blocks[f'final_0_{i}'] = SegmentationHead(decoder_channels[-1], classes, kernel_size=3)
        
        
        self.blocks = nn.ModuleDict(self.blocks)
        self.depth = len(self.in_channels)
        
        self.center = nn.Identity()
                
    def forward(self, x):        
        
        dense_x = {f"x_{i}_0": x for i, x in enumerate(self.encoder(x))}
        
        # start building dense connections
        for layer_idx in range(1, len(self.in_channels)):
            # Normal unet++ layers
            for depth_idx in range(self.depth - layer_idx):
                upsample_x = dense_x[f"x_{depth_idx+1}_{layer_idx-1}"]
                cat_x = torch.cat([dense_x[f"x_{depth_idx}_{layer}"] for layer in range(0, layer_idx)], dim=1)
                output = self.blocks[f"x_{depth_idx}_{layer_idx}"](upsample_x, cat_x)
                dense_x[f"x_{depth_idx}_{layer_idx}"] = output
            
            # Extra layers to account for using pretrained encoder 
            upsample_x = dense_x[f"x_0_{layer_idx}"]
            if layer_idx == 1:
                cat_x = None 
            else:
                cat_x = torch.cat([dense_x[f"x_0_{layer}_ex"] for layer in range(1, layer_idx)], dim=1)
            dense_x[f"x_0_{layer_idx}_ex"] = self.blocks[f"x_0_{layer_idx}_ex"](upsample_x, cat_x)
        
        out = [self.blocks[f"final_0_{i}"](dense_x[f"x_0_{i}_ex"]) for i in range(1, 5)]
                    
        return out
    
    
def unetplusplus_splitter(model):
    return L(model.encoder, model.blocks).map(params)